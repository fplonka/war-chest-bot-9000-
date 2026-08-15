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

## `value_v4_clean30` dynamics

The raw fresh-row loss spike was a curriculum wave, not divergence. At 16.4 min, `loss_new=0.0206` while `loss_old=0.00055`: newly generated targets had become much harder. About nine minutes later those rows had aged into the old replay bucket; `loss_old` peaked at `0.0109` while `loss_new` had fallen to `0.0045`. By 29.8 min the buckets converged at `0.0090/0.0082` while target SD had expanded to `0.278`. Total loss ended near `0.11` of target variance.

Most playing-strength improvement occurred from s2 through s4. The final two minutes added no measurable strength over s5. This motivates measuring generation throughput, then testing a simpler schedule over a longer ReBeL interval rather than changing several learning knobs at once.

## Performance controls

Caching the public first-context projection once per GPU wave improved the frozen-tape benchmark from `753` to `820 solves/s` (`+8.8%`). Replacing GELU with ReLU did not improve production throughput and was reverted. Reducing the training search from 64 to 32 iterations increased a controlled run from `553.5` to `653.3 solves/s` and from `436,269` to `466,622` rows in the same wall time. The 32-iteration setting therefore stays.

The optimizer ratio returns to 4. The legacy successful run used 4, while 12 trained `7.83M` optimizer rows from only `4.48M` generated rows. A ten-minute ratio-4 continuation, `value_v4_ratio4_10`, generated `1.28M` rows and trained `0.75M` rows without material optimizer debt. Its final tied its input checkpoint `273-270-57` over 600 matched games. More replay fitting did not provide measurable strength.

## Architecture comparison

The corrected full-network run, `value_v4_fullwarm30`, completed 30 minutes and generated `652,807` solves. In a 600-game direct match its final checkpoint beat Greedy `591-4-5`. The legacy `odd` final, evaluated with its own stable pre-refactor engine under the same seed and search settings, beat Greedy `567-3-30`. This anchor does not show an architecture regression.

Checkpoint-to-checkpoint results are strongly non-transitive. `value_v4_fullwarm30.final` tied its post-warm checkpoint `294-296-10`; `odd.final` also tied or slightly lost to `odd.init` (`262-274-64`) even though their Greedy results differed sharply. Elo chains and one opponent can therefore diagnose gross failure, but not rank close policies. The refactored readout is not the current blocker: deterministic full-network warm-up already produces a strong policy, and later self-play changes behavior without a stable matched-policy gain.
