# v5_aliased_l5_warm_gate

We were testing the real balanced training rate after reach aliasing, while also
repeating the safe learning setup: five minutes of Greedy value-network warm-up
followed by five minutes of ReBeL self-play. The run used two RTX 3090s, five
ordinary lanes per GPU, one retained whale lane, and true game outcomes once
games finished. The search horizon payoff annealed from 0.04 to zero during
self-play.

The Greedy phase was healthy. Held-out loss fell from 0.078 to about 0.009,
prediction spread stayed nonzero, and `snap_00.pt` was saved before self-play.
The first ReBeL game completed 41 seconds after the switch, not twelve minutes;
the warm value net and faster solver therefore supply finished-game data early.
The self-play phase completed 364,405 solves and 1,423 optimizer steps. At the
admission stop it was averaging 1,301 balanced solves/s with zero training debt.
Including the full drain and 2.3-second overrun, the final summary was 1,211.1
balanced solves/s. There were no dropped solves, exact fallbacks, or allocation
failures. Peak sampled memory was 16,313 MiB on GPU 0 and 23,901 MiB on GPU 1.

This passes the five-minute 1,200/s balanced gate even under the deliberately
unflattering drain-inclusive number. The run accumulated 3,185 completed
self-play games, so its data was not an all-draw bootstrap. `snap_01.pt` is the
final checkpoint. In the closing 30-game-per-pair ladder it beat the warm-only
checkpoint 21--1 with 8 draws, beat Greedy 21--5 with 4 draws, and beat Random
19--0 with 11 draws. The warm-only checkpoint lost to Greedy 5--14 and mostly
drew Random. The fitted ratings put the final checkpoint 278 Elo above the
warm-only checkpoint and 133 Elo above Greedy. This particular run therefore
learned to convert games rather than merely preserve draws, and it is a sound
starting point for the longer 5+25-minute run.
