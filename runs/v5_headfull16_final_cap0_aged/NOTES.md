# v5_headfull16_final_cap0_aged

We were checking the complete float16 head-data path on the steady-state
workload that matters: the learned `v5_fast16_l5_warm_gate` final checkpoint,
true-game horizon payoff zero, fixed weights, and the production two-card,
five-lane stream. The static public residual, dynamic belief embedding, first
head-GEMM output, and post-LayerNorm activation are stored as float16. The
residual addition, LayerNorm reductions, GEMM accumulation, CFR state, root
values, and training targets remain float32. No optimizer ran.

The stream completed 196,608 solves in 180.10 seconds before stopping, or
1,091.7 solves/s. The preceding build on the same nominal seed completed
178,176 at 989.0/s. The identical frozen tape is the clean speed comparison and
improved by about 10%; the live trajectories diverge after small numerical
policy changes, but this run independently shows a healthy and faster workload.
The mature 120--180 second interval ran at about 921 solves/s versus 820/s
before. There were 871 completed games, 321 large-search routes, 1,361 solver
node caps, no exact fallback, and no dropped work.

Average GPU use was 57.7% and 57.4%; peak memory was 23,225 and 22,315 MiB,
slightly below the 23,733/22,567 MiB control but still close to the 24 GiB
limit. Drain took 38.0 seconds. Both fast and precise GPU suites pass: fast
mode uses measured 0.20 synthetic-policy and 0.10 wave-composition bounds,
while zero-network, probability, root-value, exact-reuse, and precise-mode
checks remain tight. The next required gate is real Greedy-warm training plus a
checkpoint ladder; generation alone is still below the 1,400/s integration
target.
