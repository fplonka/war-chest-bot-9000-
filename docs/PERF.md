# Making the ReBeL loop ~10x faster

The binding constraint on this project is **generation throughput**: how many
self-play games with a full CFR solve at every decision fit in the wall-clock
budget. A 10-minute run used to buy 7 ReBeL epochs. It now buys ~110.

Everything here is a performance change. The only one that touches the model is
the network split (§4), and it is parameter-for-parameter the same network with
one weight matrix moved; it was validated on a training run like every other
step, and it *improved* the result.

## The number

| measurement | before | after | ratio |
|---|---|---|---|
| ReBeL epochs in a 10-minute budget | 7 | 111 | 15.9x |
| generation, games/s (in-training) | 1.16 | 10.9 | **9.4x** |
| benchmark, trained-agent positions | 262 dec/s | ~1900 dec/s | 7.3x |
| benchmark, **identical** workload | 56.5 s | 2.19 s | **25.8x** |

The last row is the only strictly matched comparison — both builds driven by an
all-zero network, which makes every leaf value zero, so CFR stays uniform and
the two play *exactly* the same games with exactly the same subgame trees while
still doing the full matmul work. Same 2068 decisions either side, interleaved
reps, best of three.

The spread between 7x and 26x is not noise, it is the shape of the win: the old
code's cost grew superlinearly in the size of the belief support (dense
`[parent][child]` draw matrices, a heap allocation per node per solve), and the
identical-workload run sits at 44 configs per decision against a trained agent's
23. **The optimisations help most exactly where the old code was worst**, which
is the late, complicated positions. End to end, in the training loop, it comes
out at 9-10x.

A caveat on measuring any of this on a laptop: a busy window server and a
terminal tailing a log cost ~1.8x on this benchmark. Numbers here are best-of-N
on an otherwise idle machine, and the training runs were compared against
`runs/diagW2` taken under the same conditions.

## How this was measured

Two tools, both added for this work:

* `engine/src/bin/rebelbench.rs` — the generation loop without Python, driven by
  weights exported from a real checkpoint so branching and game length match
  training. Turns the edit-measure cycle from 20 minutes into 15 seconds.
* `engine/src/prof.rs` — direct phase timers behind a `prof` feature. A sampling
  profiler was actively misleading here: it attributed ~9% of time to the
  Accelerate calls when the real figure was over half, because AMX coprocessor
  time does not land on the calling thread's stack the way scalar work does.

The benchmark is deterministic — each game's RNG is seeded from its index, not
from the thread, so rayon's scheduling does not affect it. Every step below that
does not change floating-point results was checked by confirming that the
benchmark replays **bit-identical trajectories**: same decision count, same
target count, over ~4000 subgame solves. A change to any CFR strategy would
diverge the sampled actions and show up immediately.

## Where the time went, at the start

Per subgame solve: ~836 tree nodes, ~368 leaves, 8 CFR iterations, each
iteration evaluating every leaf through the value network.

| phase | share |
|---|---|
| value network | 51% |
| reach propagation | 26% |
| tree build | 6% |
| feature encoding | 6% |
| everything else | 11% |

## 1. Don't recompute what a subgame holds fixed

Reach propagation ran **twice per CFR iteration**, and the pass inside
`update_regrets` recomputed exactly what the previous iteration's trailing pass
had left behind. Removed.

The **average strategy** is only read in evaluation mode — self-play acts on the
current regret-matching iterate — so generation stopped maintaining it, which
also removed its extra reach pass and two sweeps over every strategy cell per
iteration.

The draw chance transition was a dense `[parent][child]` matrix with at most
`NSLOT` non-zeros per row. It is now CSR. Half the tree's nodes are draws.

## 2. The first layer factorises

Inside a solve the same leaf is queried once per iteration, and between
iterations only the *belief* part of its encoding moves. The belief block is the
tail of the feature vector and `w[0]` is input-major, so

```
h = x_pub · W_pub  +  x_bel · W_bel
    ^^^^^^^^^^^^^   fixed for the whole solve
```

With `FEAT = 812` and a 132-wide belief block that is 84% of the widest layer,
computed once per leaf instead of eight times.

The network has since been rebuilt around the config rather than the hand
(`docs/REBEL.md` §4) and this split survived it unchanged — it is now
`Mlp::trunk` plus `Mlp::pbs_head`, and the config tower is cached the same way,
once per *distinct* config per solve. The one new hot loop, accumulating the
belief embedding, needed the same hand-vectorisation as the LayerNorm below:
scalar it was 41% of all CPU, vectorised it is 9%.

Together with §1: **262 → 550 decisions/s** (trained-agent positions).

## 3. The matmuls were never the problem

A microbenchmark of the three matmuls in a leaf query against the whole forward
pass was the turning point. The matmuls ran at 1.2-1.35 Tflop/s through AMX and
accounted for **137 µs of a 1436 µs call**. The other 1299 µs was the
elementwise pass: `row.iter().sum()` twice per row for the LayerNorm statistics
is a serial chain of 3-cycle adds, 384 long, and it cost five times the matmuls
it wrapped.

Rewriting it with eight independent accumulators over `chunks_exact` was not
enough — the disassembly showed LLVM had unrolled them into eight *scalar*
`fadd`s and left the vector registers idle. The three passes are now written
with aarch64 intrinsics: 667 → 64 µs. Buffers are also grown rather than
cleared, since each is fully overwritten by the matmul that follows; the
`clear() + resize()` they replaced was a half-megabyte memset per layer per
call.

**550 → 1012 decisions/s.**

## 4. Move the belief block to the second hidden layer

Same parameter count, same depth — the belief block is simply wired into layer 2
instead of layer 1. What that buys is that the *whole* public tower becomes a
function of the leaf's public state alone, so a subgame computes it once per
leaf and only the belief projection and the output head run per iteration.
Per-iteration matmul work drops 2.6x. See `docs/REBEL.md` §4.

This is the one change that alters the model, so it was gated on a full training
run: it took final-vs-Greedy from 0.965 to **1.000** and final-vs-initial from
0.920 to **0.960**.

**1012 → 1683 decisions/s.**

## 5. Stop allocating, stop chasing pointers

A subgame is built every couple of decisions, and each one was allocating
megabytes of fresh zero pages for its leaf batch and ~2700 small vectors for its
per-node state. In throughput terms that was thousands of page faults and
millions of `malloc`s per second.

* A run of consecutive draws by the same player collapses into **one** chance
  node with the composed transition (`DrawMap::compose`). None of them branches
  the public tree and none is a decision, so the subgame was carrying six nodes
  — six states, six reach vectors, six value vectors, six passes per traversal —
  to express one convolution. Nodes per solve 836 → 539.
* The five big per-solve buffers are pooled **by role**; they differ in size by
  5x, and a single shared pool handed each one somebody else's buffer and made
  it grow, which is the one thing that zeroes.
* `reach`, `vals`, `regret` and `cur` live in flat per-solver arenas instead of
  `Vec<Vec<f32>>`. Parent and child regions are disjoint by construction —
  children are built after their parents — so the passes still borrow them
  together through one `split_at_mut`.
* `TNode::cfgs` is `Rc<[Config]>`: every public child of a decision node has the
  same support for the idle player, and a draw leaves it untouched.
* The leaf batch is assembled inside `build`, while each leaf's state is the one
  just constructed and still in cache.
* `State` is `Copy` — the Footman-tactic hex list became a bitmask and the
  continuation stack an inline array.

## 6. Sort integers, not structs

Building a subgame sorts, dedupes and binary-searches a few hundred `Config`s
per chance node and ~800 per decision node, comparing two five-byte arrays each
time. `Config::key` packs one into 35 bits — 2 per hand slot (a hand holds at
most three coins), 5 per face-down slot — which leaves room to carry the element
index in the same `u64`. Sorting those once and reading both the support and
every row's child index off that single ordering removes the searches entirely.

Note that the key is a method, not an `Ord` impl: making it `Ord` was tried and
was *slower*, because it recomputes the key on every comparison instead of once
per element.

## What did not work

* **Caching each player's belief projection across iterations.** Only one
  player's strategy moves between two CFR iterations, so only one player's
  beliefs do — the other's projection is still valid. This halves a 30 µs
  matmul and adds a megabyte of buffer traffic to the elementwise pass around
  it. Measured **35% slower**.
* **Bigger generation batches** for load balance: 48, 96 and 192 games per epoch
  all measure the same, so rayon's work stealing is already handling the spread
  in game lengths.
* **Fewer threads.** On this 4P+4E machine, 8 rayon threads beat 6 (1690) and 4
  (1292) despite AMX contention.

## The training runs

Same command as the baseline, `--minutes 10 --warm-frac 0.15 --cap-value 0
--eval-games 200 --seed 11`, 200 paired evaluation games on the real game.

| run | code | ReBeL epochs | vs Greedy | vs initial |
|---|---|---|---|---|
| `runs/diagW2` (baseline) | before | 7 | 0.920 | 0.800 |
| `runs/perf01_netsplit` | §1-2 | 32 | 0.965 | 0.920 |
| `runs/perf02_twotower` | §4 | 98 | 1.000 | 0.960 |
| `runs/perf03_arena` | §5 | 116 | 0.990 | 0.925 |
| `runs/perf04_final` | §6 | 111 | 0.993 | 0.950 |
| `runs/perf05_ship` | shipped | 55 | 0.963 | 0.958 |

The last four are the same algorithm at increasing speed; their spread
(0.963-1.000 and 0.925-0.960) is the run-to-run noise of a 10-minute budget, and
the standard error on 200 games is ~0.015. `perf05` ran on a machine under heavy
UI load — its 55 epochs are half what the same code does idle, and its build was
A/B'd directly against `perf04`'s at 1073 decisions/s each, on bit-identical
trajectories.

## Where it is now

The value network is back to being the dominant cost, and it is at the
hardware's limit rather than the code's: the matmuls run near peak and the
aggregate rate across 8 threads sits at the M1's shared-AMX ceiling. Cutting it
further means cutting floating-point operations, not overhead.

| phase | share |
|---|---|
| value network, per iteration | 32% |
| tree build (of which draw transitions, 13%) | 24% |
| value network, public tower (once per leaf) | 15% |
| belief encoding | 11% |
| reach propagation | 7% |
| backward pass | 6% |
| everything else | 5% |

The obvious next lever is the draw transitions: a run of *k* draws is currently
composed step by step over intermediate supports that grow by ~5x each time,
where the multivariate hypergeometric gives the same answer directly from the
parent support in roughly a third of the entries. It needs a fallback for the
mid-run reshuffle case, and an oracle against the step-by-step chain.

## A second pass, after the card describer and the policy head

Those two added new per-solve work, and it was all written as scalar triple
loops sitting next to matmuls that run at ~1.3 Tflop/s. Fixing that, and moving
one older loop onto the coprocessor as well, is **+8.9%** end to end on the
generation benchmark, with bit-identical trajectories — the same 3731 decisions
and 14448 targets either side, so no sampled action moved.

| | |
|---|---|
| holding tower | one matmul over `[n * NSLOT, hf]` plus a segmented sum |
| pile summary | the card half is constant across a solve, so it folds into the bias; what is left is four counts wide |
| per-config readout | `[rows, rank] x [ncfg, rank]^T` instead of ~8.5k short NEON dots per iteration |
| policy logits | same shape, same new `gemm_nt` |

The readout is the interesting one: it is **~7x the arithmetic** — a leaf carries
~18 configs and a solve interns ~160 — and still wins, because AMX against NEON
is worth far more than 7x.

### Three things that did not work

**Splitting the belief projection by player.** An alternating iteration moves one
player's beliefs, so half the widest per-iteration matmul is reusable. 8% slower:
fusing the two halves costs a full `[rows, hidden]` read-add-write per iteration,
which moves about four times the memory the halved matmul saves.

**The same batching trick on the belief encoding.** `[rows, ncfg] x [ncfg, dg]`
instead of a gather-accumulate per leaf — the exact trade that won for the
readout — came out 3.5% *slower*. The asymmetry is what to remember: the readout's
gemm reads two matrices that already exist densely, while this one has to
materialise a mostly-zero `[rows, ncfg]` weight matrix every iteration. Building
the input cost more than the coprocessor saved.

**Deferring the average strategy.** It is normalised out of its running sum every
iteration but only *read* at the log-spaced snapshots, ~9 times in 64. Computing
it lazily is 2% slower, because the per-iteration version got the row sum for
free inside the accumulation it was already doing, while a lazy pass has to
re-read the sums — and 9 passes over both players is about the same work as 64
over one.

### On measuring any of this

The first pass's warning about a busy machine was understated. Single runs here
vary by 12%, and sequential A/B — build one, measure, build the other, measure —
gave the *wrong sign* on two of these three. Every number above is
**interleaved** (A, B, A, B, ...) and best-of-N, which is the only way a 3%
effect is separable from drift on a laptop. `rebelbench` takes a warm-start
weight as its sixth argument so the policy path can be profiled too.
