# warmctl

Control arm of the warm-ratio A/B: the row-unit `train_gen_ratio = 0.5` from
`1a85af8`, which trains the greedy warm phase eightfold weaker than the ReBeL
phase at the same setting, because greedy retains one row per solve while
TurboReBeL retains about eight.

**807.7 solves/s, debt 986, zero dropped, 9.6 s overrun.** Warm loss entering
ReBeL was 0.0478 against the fixed arm's 0.0094.

The fixed arm `warmfix` (4.0 optimizer rows per solve in the warm phase) is
the other half of this pair; `docs/GPU_PERF_GOAL.md` has the numbers and why
the warm phase's fit is a first-order throughput term.
