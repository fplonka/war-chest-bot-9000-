# Value v5 — the architecture rebuild

## What was wrong with v4

v4 spent its compute in the wrong place, and the accounting is not close. CFR
re-asks every leaf on every iteration, so at depth two, `T=64` a solve makes
~2,030 board evaluations against ~158,000 belief-conditioned ones. v4 put a
three-layer 384-wide flat MLP on the board and two 384-wide LayerNorm layers on
the per-iteration path:

| | v4 | v5 |
|---|---:|---:|
| board encoder | `1.7%` of network MAC | **`63%`** |
| per-iteration path | `96.4%` | `33%` |
| total | `58.0` GMAC/solve | `48.9` |
| projected throughput | `522` solves/s (measured) | `~620` |
| parameters | `1.55M` | `0.95M` |

DeepStack, ReBeL, TurboReBeL and Student of Games all avoid this by making the
infoset an **output index**: one tower per public leaf emits every infoset
value as one row of the output matrix, so an infoset costs 500-1536 MAC. v4
charged 33,000 MAC of GEMM plus a GELU per config. We cannot table the output
rows — the config set is variable, median 22 and p99 567 — so v5 *generates*
them from a config encoder and reads out with a dot product. That is the only
structural novelty, and it is the standard open-vocabulary output layer.

## The shape

```
public state ─► TRUNK (8 hex residual blocks, global pooling) ─► P   once/leaf
config c ─────► CONFIG ENCODER ─► f(c) readout, g(c) pooling         once/config
     [Σβ_own g, Σβ_opp g] ─► JOIN (3 blocks, 128 wide) ─► h          every iteration
                        v(c) = <f(c), h> + bias
```

Widths: `TYPE=64, C=128, BLOCKS=8, D=256, POOL=64, CFGH=128, JW=128,
JBLOCKS=3`. The trunk is KataGo-shaped — pre-activation residual blocks over
the board's own hex adjacency with a global-pooling bias in *every* block
(pooling is KataGo's second-largest measured ablation at `1.60x`, and it costs
`1.8%` of a block). `join_p(P)` is cached once per solve, which is what lets
the board vector be wide.

### The central bet to test

Unlike DeepStack, ReBeL and Student of Games, v5 keeps the trunk public-only
and injects both beliefs in the join. This is deliberate: at depth two the
public trunk runs about 2,030 times per solve while the belief-conditioned path
runs about 158,000 times. It is also an expressivity tradeoff. The hex features
cannot be recomputed after learning that the opponent probably lacks a given
coin. Treat this split as an experiment, not settled literature. The first
architecture ablation should move one narrow hex block after belief injection
and measure strength at matched solver throughput.

## Slot-permutation equivariance, and the bug it caught

Which slot index the draft assigns a unit is a pure relabelling, so the network
must be exactly invariant to permuting each player's five slots. v4 was not:
`value_net.py:100` flattened the ten `[pile counts ‖ token]` blocks in slot
order into a dense layer, so the same unit got different weights depending on
the draft. That partly undid the card describer built to avoid exactly this.

v5 treats the ten coin types as a set of tokens described by their printed card
facts, with no unit-identity embedding at all. The first draft of v5 still
failed the check — measured `2.1e-2` against a value spread of `4.2e-2`, half
the signal — because the belief's raw count marginals were fed to the join as
30 numbers through a dense layer with per-slot weights.

The fix removes them and makes the pooling vector carry them instead:

    g(c) = cfg_g(u(c)) + Σ_k Σ_{zone} count[c][k][zone] · V[k][zone]

with `V[k] = cfg_m(card_k)` a per-zone embedding of the *card*. `cfg_m` is
linear in the counts, so `Σ_c β(c) g(c)` carries the belief's exact expected
holding of every card, bound to that card — "they almost certainly cannot play
an Archer this turn" arrives as a marginal rather than as an average of GELUs.
`V` depends only on the card table, so it costs two rows per solve and fifteen
accumulations per config. The join input shrank from 158 to 128 in the process.

With that, permutation error is `7.5e-8` against a `4.2e-2` spread — exact to
float32. TurboReBeL's largest architecture-free win was its `24x` isomorphic
augmentation; this is the same kind of symmetry, taken by construction instead
of by augmentation.

## Auxiliary head

Per location hex, a 3-way logit over the final owner of that location at the
end of the game the row came from. This is Go ownership, which is KataGo's
largest measured ablation (`1.65x` with score), and it is a genuine
decomposition of the outcome here: you win by getting all six markers down.
Training only — it is not in the weight blob and the engine never runs it.

## Deliberately not done

* **Zero-sum enforcement.** Three v4 attempts lost. `odd` showed the
  reparameterised readout learns fine (residual `0.0006`, ladder `+578`) and
  died on throughput (`844` against `1081` solves/s). The game is zero-sum; a
  correct architecture should learn that, and forcing it has only ever cost
  strength here. Left as a diagnostic.
* **Policy head.** The v4 measurement (`arch_policy`, `646` against `663` and
  `696`) tested it as an auxiliary loss only; the warm-start run that was its
  actual justification was killed before producing a number. Worth revisiting
  as a search change, not as a loss term.
* **Direction-aware message passing.** The trunk sums its six neighbours with
  one shared weight, so it sees adjacency and distance but not *heading*.
  Crossbowman and Lancer need a straight line, and lines of advance matter
  generally. Per-direction taps cost `2.7x` the trunk; three axis pairs cost
  `1.67x` and would still express a line. This is the first trunk experiment to
  run once the box is free.


# Value v4 runs

## Main finding

The failed warm starts mixed two incompatible labels. With `eval_mix=0.5`, each Greedy target was half deterministic static evaluation and half one sampled game outcome. During `public_only` warm-up the network had no private information that could explain the sampled outcome. It therefore fit high-variance label noise and entered ReBeL with a poor value scale. `eval_mix=1.0` makes the warm label a deterministic function of the saved public state. The same architecture then fit a fresh matched-generator holdout (`RMS 0.0081`, correlation `0.9998`) and trained successfully.

This setting affects Greedy warm-up only. ReBeL continues to use its bootstrapped counterfactual targets.

## Runs

| Run | Change | Result |
|---|---|---|
| `value_v4_beliefw` | Belief-weighted loss; 10 min, 3 min warm, 4 optimizer rows/solve | No clear improvement over the broken baseline. Final target SD `0.193`; balanced generation `228 solves/s`. |
| `value_v4_public_warm3` | Public-only warm-up with private paths zeroed; `eval_mix=0.5` | Training loss fell, but fresh-state behavior remained poor because the warm labels still contained sampled-outcome noise. |
| `value_v4_public_warm30` | Same, plus cached public projection | Stopped early. Cached projection improved the frozen-tape kernel benchmark from about `753` to `820 solves/s`, but the run still used the bad warm labels and too few optimizer updates. |
| `value_v4_public_warm30_r12` | 12 optimizer rows/solve | Stopped early after the initial instability. More fitting cannot repair an unlearnable warm target. |
| `value_v4_warm5_ratio1` | Five-minute warm-only control with `eval_mix=0.5` | Replay fit improved while a matched fresh-generator holdout remained poor (`RMS 0.509`, correlation `0.455`). This isolated label noise/data efficiency rather than optimizer capacity. |
| `value_v4_warm3_clean` | Three-minute warm-only control with `eval_mix=1.0` | Decisive control: matched holdout `RMS 0.0081`, correlation `0.9998`. |
| `value_v4_rebel_clean3b` | Start ReBeL from the clean warm checkpoint; 12 optimizer rows/solve | Stable three-minute bootstrap check. Target mean stayed small and loss fell to `0.00027`. |
| `value_v4_clean30` | Clean end-to-end run; 3 min deterministic public warm, then ReBeL; 12 optimizer rows/solve | Healthy learning. Ladder Elo versus Greedy: init `-120`, s1 `-94`, s2 `+83`, s3 `+268`, s4 `+393`, s5 `+425`, final `+431`. Final beat Greedy `19-0-1`; final and s5 were tied within noise (`39-35-6`). |
| `value_v4_depth1_precise30` | Production defaults: depth 1, 32 iterations, 4 optimizer rows/solve, and fused context normalization | Verified 30-minute run: `10,835,186` solves, `7,225` raw and balanced solves/s including drain, no dropped jobs, and no exact fallbacks. The ladder improved at every checkpoint: init `-91`, s1 `+396`, s2 `+499`, s3 `+524`, final `+562` Elo versus Greedy. |

## `value_v4_clean30` dynamics

The raw fresh-row loss spike was a curriculum wave, not divergence. At 16.4 min, `loss_new=0.0206` while `loss_old=0.00055`: newly generated targets had become much harder. About nine minutes later those rows had aged into the old replay bucket; `loss_old` peaked at `0.0109` while `loss_new` had fallen to `0.0045`. By 29.8 min the buckets converged at `0.0090/0.0082` while target SD had expanded to `0.278`. Total loss ended near `0.11` of target variance.

Most playing-strength improvement occurred from s2 through s4. The final two minutes added no measurable strength over s5. This motivates measuring generation throughput, then testing a simpler schedule over a longer ReBeL interval rather than changing several learning knobs at once.

## Performance controls

Caching the public first-context projection once per GPU wave improved the frozen-tape benchmark from `753` to `820 solves/s` (`+8.8%`). Replacing GELU with ReLU did not improve production throughput and was reverted. Reducing the training search from 64 to 32 iterations increased a controlled run from `553.5` to `653.3 solves/s` and from `436,269` to `466,622` rows in the same wall time. The 32-iteration setting therefore stays.

The optimizer ratio returns to 4. The legacy successful run used 4, while 12 trained `7.83M` optimizer rows from only `4.48M` generated rows. A ten-minute ratio-4 continuation, `value_v4_ratio4_10`, generated `1.28M` rows and trained `0.75M` rows without material optimizer debt. Its final tied its input checkpoint `273-270-57` over 600 matched games. More replay fitting did not provide measurable strength.

The production configuration exceeds the `800 solves/s` requirement by `9.0x` over a full 30-minute run. It generated `75,136,466` replay rows from `10,835,186` solves, admitted only three oversize routes, and dropped no work. The independently evaluated checkpoints strengthen monotonically from 5 through 29 minutes; the final beat the 23-minute checkpoint with an aggregate score of `0.545` over 200 games.

## Depth-two, 64-iteration DCFR throughput

A deep solve is a different machine from a depth-one solve. At steady state a
depth-two subgame carries `2,181` leaf rows against roughly `180` for the
shallow default, and the head runs `78` times per solve: `64` CFR iterations
plus one fixed-policy pass per player for each of the seven carried root
beliefs that `snapshot_iters(64)` produces. That is `1.17e11` FLOP of head GEMM
per solve, so `800 solves/s` needs `93.7 TFLOP/s` sustained. Two RTX 3090s
offer `71.2 TFLOP/s` of FP32, which is why the FP32 path cannot reach the
target and the tensor cores are not optional.

Controlled four-minute runs, depth 2, 64 DCFR iterations, seed 101, 32 actors
per worker so games complete and the solve-size mixture settles:

| build | solves/s | solves | exclusive routes |
|---|---:|---:|---:|
| before | `294.9` | `70,761` | `44` |
| chunked public tower and kernel traffic | `328.0` | `78,716` | `0` |
| tensor-core head GEMMs | `531.6` | `127,565` | `0` |

The public tower is a per-row map, so it now runs in row chunks. Its input row
is `PUBLIC_IN` floats wide and sizing that buffer for a whole wave made a
mature wave reserve over four gibibytes, which routed it to the exclusive
one-job lane; chunking removed every such route and the redundant device copy
that used to cache the public context projection. LayerNorm now holds its row
in registers across the mean and variance passes instead of reloading it from
the arena twice, which brought `context_norm_gelu` and `norm_gelu` to `937
GB/s` -- DRAM peak, so they are done. The readout keeps the context row with
the joint bias folded in, and the output weights, in shared memory, reads
candidates as `float4`, and values four leaves per block so that neighbouring
leaves share their candidate pool through L1. The sweeps name their arena and
table bases once instead of reloading them through `w` after every store.

With the tensor-core GEMMs the mix is `40.4%` GEMM, `12.8%`
`context_norm_gelu`, `11.0%` readout, `9.8%` `norm_gelu`, `8.1%`
`belief_sums`, `7.6%` reach sweep, `6.7%` backprop sweep.

Tiling the readout exposed a race that predated it. `belief_sums` parks the
opponent reach mass in the value slot the readout is about to overwrite, and
the readout read it with no barrier between the warps that read and the warps
that write. Whether a leaf got the reach mass or a leaf value depended on warp
timing; the reuse-invariance check caught it intermittently, at up to `0.30`
absolute in a single action probability. The read now happens before the
block's one barrier.

Holding the candidate, belief and public-context embeddings as halves was tried
and reverted: `522.5` against `531.6 solves/s`. The readout is instruction-bound
on GELU, not byte-bound -- `tanhf` already compiles to one `tanh.approx.f32`, so
`96` of its `130` inner instructions are the nonlinearity -- and the extra
`__half22float2` conversions cost more than the halved loads saved. Only
`context_norm_gelu` was genuinely at DRAM peak, and it holds one of three
streams. Precision was not worth spending there.

The oracle contract splits along the same line. `Blas` carries the math mode, so
the three CPU-oracle tests build a precise executor and keep their original
tolerances against exact math, while `tensor_core_head_tracks_exact_math` runs
the production path and bounds it against that same exact solve: root values,
which are what training consumes, agree to `1.9e-4`, and a regret-matched action
probability -- a ratio of differences between leaf values, so two orders more
sensitive -- to `4.8e-2`.

The verified 30-minute run, `value_v4_d2i64dcfr_30`, sustained `522.2 solves/s`
at production defaults: `783,039` solves, `3,131,392` optimizer rows, `14,161`
games, no exclusive route, no dropped work, no exact fallback. DCFR trains well
at this depth -- ladder Elo against Greedy went init `-49`, s1 `+193`, s2 `+498`,
s3 `+638`, final `+681`, and the final beat Greedy `20-0-0` in its seeding pair.
That is `1.77x` the pre-optimization rate and `65%` of the `800 solves/s` target.

CUDA Graphs and larger waves were both tested and rejected: graph capture cannot
reuse an executable across the shapes a live stream produces, so it measured
`466.1` against `531.6 solves/s`. Reaching `800` from here needs the non-GEMM
half cut by about `2.4x`. The two places that can give it are the LayerNorms,
which are `22.6%` of GPU time and sit exactly at DRAM peak in FP32 -- half
storage halves them -- and the reach and backprop sweeps at `14.3%`, which walk
their reverse-gather rows one thread per row and so read uncoalesced.

## The from-scratch depth-two collapse

The bootstrap collapse was a recipe defect, not a search or estimator defect,
and the variable is the size of the live game set. Four seeds, seven minutes
each, five of them warm, everything else identical:

| seed | live 1152 (spread, games) | live 4608 (spread, games) |
|---|---|---|
| 95 | `0.097` -> `0.136`, 61-185 | `0.001`, 0 |
| 96 | `0.104` -> `0.165`, 94-263 | `0.049` -> `0.056`, 0 |
| 97 | `0.199` -> `0.241`, 131-245 | `0.001`, 0 |
| 98 | `0.102` -> `0.144`, 46-220 | `0.067` -> `0.077`, 0 |

Four of four train at `1152`; none of four train at `4608`, and not one of them
completes a single game in any ten-second window. That is the mechanism. A
depth-two solve costs about `550/s` against `7,900/s` at depth one, so with
`4608` games resident each one advances `0.12` decisions a second and a
hundred-decision game needs some `19` minutes to finish. Until a game finishes,
no terminal outcome has entered the replay buffer, so every label the network
trains on was produced by the network itself. `network -> CFR -> network` has no
shortage of fixed points when nothing anchors it, and any constant is one; the
runs at `0.049` and `0.067` are caught drifting monotonically toward theirs.

This also explains the seed lottery it was mistaken for. The `19` minutes are
comparable to a thirty-minute run, so a long run does eventually start finishing
games -- `value_v4_d2i64dcfr_30` finished `14,161` of them -- but only after its
target scale has already degenerated, which is why that run ends up *below* its
own initialisation (init `544`, six minutes `407`, thirty minutes `499`). Short
runs and short-lived seeds never get there at all. Sizing the live set removes
the lottery rather than improving the odds.

The fix is one knob instead of two. A worker now holds exactly as many live
games as it can have solves in flight (`gpu_inflight`), because a game held
beyond that can never be worked on sooner and only takes longer to finish;
`gpu_actors` is deleted. It is also faster, not a trade: `596 solves/s` against
the `390-522` the oversized set managed, since games that finish concentrate
beliefs and keep subsequent solves small.

Two hypotheses recorded in `TODO.md` were measured and both are refuted. The
value target is not the wrong estimator: we implement TurboReBeL, whose Phase 2
(Algorithm 2, line 13) specifies exactly `v^s(b) <- UpdateCFV(S', s)`, the
backpropagation under the fixed final reference strategy that `value_under`
performs, and that Phase 2 is what earns the `T+1` rows per decision
`selfplay_walk.rs` pins. The zero-reach uniform fallback never fires: over a
depth-two, 64-iteration solve under a flat network -- precisely the warm-start
condition it was suspected to spoil -- no reach among `114,585` rows is zero or
even below `1e-30`, and no strategy sum among `64,822` is zero. The reference's
`1e-80` floor is not representable in an `f32` arena in any case.

Half-precision head *activations* were then tried and reverted, and this one is
worth keeping written down. Holding Xb, H and H2 as halves -- feeding the tensor
cores their operands in the form they already multiply in -- is worth `+12.7%`,
`599.1` against `531.6 solves/s`, and it passes every oracle: root values still
track the exact CPU solve to `1.9e-4`. It nevertheless destroys the ReBeL
bootstrap. From a five-minute warm start the target spread collapses from
`0.091` to `0.001` within two ReBeL epochs and never recovers, at both seeds
tried, where the FP32 build at seed 95 does not. A warm network's within-support value spread is a few percent of its
across-query spread, so the differences CFR needs are near the half grid at the
start of the bootstrap, and quantising them away leaves the search nothing to
differentiate. Starting from an already-trained checkpoint hides this completely,
which is why the four-minute controls looked healthy. The lesson generalises: a
solve accuracy bound measured against a converged network says nothing about
whether the bootstrap can get there.

## The verified depth-two run

`d2t64_fixed_30`, thirty minutes from scratch at seed 95 with the live set
sized to inflight, is the first depth-two bootstrap here that learns
monotonically: `861,850` solves, `19,022` completed games, `horizon=0.01`, so
the games end in real outcomes rather than at the draw cap.

| checkpoint | trained | Elo vs Greedy |
|---|---:|---:|
| init | 5min | `-86` |
| s1 | 11min | `+373` |
| s2 | 17min | `+474` |
| s3 | 23min | `+563` |
| final | 29min | `+611` |

Against `value_v4_d2i64dcfr_30`, the old-recipe thirty-minute run at the same
seed, depth and iteration count, its final won `125-70-5` over 200 games --
score `0.637`, `+98` Elo, `z=3.9`, `p=1e-4`. Against the pre-refactor
`traverser` final on the cross-engine relay at matched search it won
`130-66-4`: score `0.66`, `+115.2` Elo, `p=5.7e-6` decisive and `1.9e-5` on
the colour-swapped pairs. Sustained throughput rose from `522.0` to
`574.6 solves/s`; the `850-988/s` of the early windows is not sustainable,
since trees grow as the policy sharpens.

What this does *not* show is depth two beating depth one at thirty minutes. It
does not: `value_v4_depth1_precise30` scores `+386.6` against the same
traverser where this run scores `+115.2`, because in the same wall-clock it
buys `10,835,186` solves against `861,850` -- a factor of `12.6`. The case for
depth two rests on the target quality per solve paying for that factor over a
run long enough to show a crossover, and thirty minutes is not that run. What
is settled is that depth two now trains at all, reliably, and that its own
curve no longer bends downward.

## Architecture comparison

Against the `traverser` final -- the last checkpoint of the pre-refactor
parameterised architecture, run on its own engine revision through the
cross-engine relay at matched search (depth 2, 64 iterations, colour-swapped
pairs, every state cross-checked between the two engines):

| new checkpoint | W-L-D | score | Elo | paired | paired p |
|---|---:|---:|---:|---:|---:|
| `value_v4_depth1_precise30.final` | `178-17-5` | `0.9025` | `+386.6` | `84-3-13` | `1.4e-21` |
| `value_v4_d2i64dcfr_final.final` | `177-18-5` | `0.8975` | `+376.9` | `82-0-18` | `4.1e-25` |

The first row is the fair comparison: both sides are thirty minutes from scratch
on their own revision. The second starts from the first, so it measures the deep
run's checkpoint, not thirty minutes of deep training from nothing -- and it is
slightly *behind* its own initialisation, which the run's own ladder also shows
(init `544`, six minutes `407`, thirty minutes `499`). Training on depth-two
targets moves the network off the depth-one optimum and `828k` solves is not
enough to get back.


The corrected full-network run, `value_v4_fullwarm30`, completed 30 minutes and generated `652,807` solves. In a 600-game direct match its final checkpoint beat Greedy `591-4-5`. The legacy `odd` final, evaluated with its own stable pre-refactor engine under the same seed and search settings, beat Greedy `567-3-30`. This anchor does not show an architecture regression.

Checkpoint-to-checkpoint results are strongly non-transitive. `value_v4_fullwarm30.final` tied its post-warm checkpoint `294-296-10`; `odd.final` also tied or slightly lost to `odd.init` (`262-274-64`) even though their Greedy results differed sharply. Elo chains and one opponent can therefore diagnose gross failure, but not rank close policies. The refactored readout is not the current blocker: deterministic full-network warm-up already produces a strong policy, and later self-play changes behavior without a stable matched-policy gain.

The direct cross-engine ladder removes the Greedy anchor. The current final beat
`odd.final` 349-218-33 over 600 games (score `0.6092`, `+77.1` Elo). The
colour-swapped pair test was 116-48-136 (`p=1.12e-7`, two-sided), and each of
three independent 200-game shards favored the current checkpoint. This rules
out an architecture regression in the trained stack. It does not isolate
architecture from training trajectory; that would require matched-data
training runs for both readouts.

## The train-to-generate ratio

Four twenty-minute runs at seed 95, depth two, 64 iterations, identical but for
`train_gen_ratio`, then their finals played directly against each other rather
than compared through separate Greedy-anchored ladders, which this file has
already found cannot rank close policies:

| ratio | solves | optimizer rows | balanced | games | direct score vs `4` |
|---|---:|---:|---:|---:|---:|
| 2 | `582,367` | `1,164,288` | `647.0/s` | `10,058` | `0.302` over 200 |
| **4** | `577,507` | `2,309,120` | `641.5/s` | `12,550` | -- |
| 8 | `491,968` | `3,935,232` | `546.8/s` | `7,818` | `0.150` over 20 |
| 16 | `367,542` | `5,879,808` | `408.4/s` | `6,312` | `0.050` over 20 |

Joint ratings put `4` at `731` against `2` at `587`, and `930` against `643` for
`16` and `541` for `8`. The margins are `p=2e-8`, `4e-5` and `2.6e-3`. Four wins,
and the reason it wins is legible in the two columns beside the result.

Generation saturates near `645 solves/s`. Ratios `2` and `4` both reach it --
`582,367` and `577,507` solves are the same number -- so dropping to `2` buys no
extra data and merely halves the passes taken over data already in hand. Above
`4` the throttle engages the other way: the optimizer holds generation back to
keep its ratio, and `16` pays `36%` of its solves for passes over rows the
network has already moved past. Since ReBeL's targets are induced by the current
network, re-fitting stale rows harder is chasing a distribution that has left.

So the right ratio is the largest one that does not throttle generation, and
that is identifiable without playing a single game: run at the ceiling and read
`debt` -- at `4` it is `908` rows against `2,309,120`, four hundredths of one
percent, meaning the optimizer keeps pace exactly. The criterion travels. If
kernels get faster or solves get deeper the ceiling moves, and the same
debt-at-ceiling reading will name the new ratio; `4` is the answer for depth two
at `T=64` on two 3090s, which is where the default already sat.

## Depth one against depth two, and what the Greedy anchor hides

At matched search -- both sides playing depth two, 64 iterations, through the
relay with colour-swapped pairs -- the depth-one thirty-minute run beats the
depth-two thirty-minute run `139-56-5`. That is `0.2925` to depth two,
`-153.4` Elo, `p=2.5e-9`, and `51-7-42` on the pairs. The two runs are
otherwise the same experiment: thirty minutes from scratch, seed 95, same
architecture, differing in the recipe they train under (`depth 1/T=32` against
`depth 2/T=64`).

The per-run ladders say the opposite. On its own Greedy-anchored ladder the
depth-two run finishes at `+610.7` and the depth-one run at `+561.8` -- depth
two ahead by `49` where direct play puts it behind by `153`. This is worse than
the imprecision already noted in this file: the anchor *misorders* them. Both
nets beat Greedy far too easily for the margin to carry information, so the
rating saturates and what is left is noise around a ceiling. Run reports and
ladder Elo can show that a run learned; they cannot choose between two recipes.
Every recipe decision here was settled by direct play for that reason.

Depth two was still improving at thirty minutes, so the question is whether it
crosses over later. Measured against a fixed yardstick -- the depth-one final,
matched search, 200 games each -- it does not look like it:

| depth-two checkpoint | trained | Elo vs the depth-one final |
|---|---:|---:|
| s2 | 17min | `-354.5` |
| s3 | 23min | `-217.3` |
| final | 29min | `-153.4` |

The gap closes by `+137.2` and then `+63.9` per six minutes. Reading the first
of those alone predicts a crossover at about `43` minutes; the second says the
approach is geometric, decaying by `0.466` every six minutes, `tau = 7.9min`,
with an asymptote `97.7` Elo *short* of the depth-one final -- and the depth-one
final is itself only a thirty-minute checkpoint that would keep moving. Three
points fitted across the tail of a bootstrap transient is a weak extrapolation
and does not prove an asymptote, but it does rule out the near-term crossover,
which is the claim that would have justified depth two for the long run.

So the recipe for a long run is depth one on this evidence, and the reason is
arithmetic rather than subtle: depth two buys better targets at `12.6x` the
price per target, and nothing measured here shows the quality repaying the
count. The case for depth two now needs a run long enough for depth one to
actually saturate against a fixed opponent, which thirty minutes is not.
