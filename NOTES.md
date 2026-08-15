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

## Architecture comparison

The corrected full-network run, `value_v4_fullwarm30`, completed 30 minutes and generated `652,807` solves. In a 600-game direct match its final checkpoint beat Greedy `591-4-5`. The legacy `odd` final, evaluated with its own stable pre-refactor engine under the same seed and search settings, beat Greedy `567-3-30`. This anchor does not show an architecture regression.

Checkpoint-to-checkpoint results are strongly non-transitive. `value_v4_fullwarm30.final` tied its post-warm checkpoint `294-296-10`; `odd.final` also tied or slightly lost to `odd.init` (`262-274-64`) even though their Greedy results differed sharply. Elo chains and one opponent can therefore diagnose gross failure, but not rank close policies. The refactored readout is not the current blocker: deterministic full-network warm-up already produces a strong policy, and later self-play changes behavior without a stable matched-policy gain.

The direct cross-engine ladder removes the Greedy anchor. The current final beat
`odd.final` 349-218-33 over 600 games (score `0.6092`, `+77.1` Elo). The
colour-swapped pair test was 116-48-136 (`p=1.12e-7`, two-sided), and each of
three independent 200-game shards favored the current checkpoint. This rules
out an architecture regression in the trained stack. It does not isolate
architecture from training trajectory; that would require matched-data
training runs for both readouts.
