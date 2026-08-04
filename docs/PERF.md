# Making the ReBeL loop ~10x faster

The binding constraint on this project is **generation throughput**: how many
self-play games with a full CFR solve at every decision fit in the wall-clock
budget. A 10-minute run used to buy 7 ReBeL epochs. It now buys ~120.

Everything here is a performance change. The only one that touches the model is
the network split (§4), and it is parameter-for-parameter the same network with
one weight matrix moved; it was validated on a training run like every other
step, and it *improved* the result.

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
computed once per leaf instead of eight times. `forward_split` also emits only
the output head the current traversal reads.

Together with §1: **262 → 550 decisions/s.**

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
  it. Measured **35% slower**. `Mlp::forward_split` says so.
* **`reserve - E[hand] - E[facedown]` for the bag composition.** Algebraically
  equal to averaging each config's own bag and half the arithmetic, in a loop
  that runs once per leaf per player per iteration. It is also numerically wrong
  exactly where it matters: when a player's whole reserve is in hand and face
  down, every config's bag is empty and the composition must stay zero, but the
  subtraction leaves a ~1e-7 residue per slot which then *normalises to one*.
  Caught by `belief_block_matches_the_direct_definition`, which was written for
  this change and now stays as a permanent oracle.
* **Bigger generation batches** for load balance: 48, 96 and 192 games per epoch
  all measure the same, so rayon's work stealing is already handling the spread
  in game lengths.
* **Fewer threads.** On this 4P+4E machine, 8 rayon threads beat 6 (1690) and 4
  (1292) despite AMX contention.

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
