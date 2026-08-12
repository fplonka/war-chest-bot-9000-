# warmfix

The fixed arm of the warm-ratio A/B: same build and schedule as `warmctl`
(5 min Greedy warm + 5 min ReBeL, both cards, trainer on card 1, seed 1),
differing only in that the warm phase trained at 4 optimizer rows per solve
instead of 0.5.

**1,289.0 solves/s (1,288.9 balanced), debt 112 of the 1,024 the goal allows,
zero dropped, one oversized route, 9.6 s overrun.** The warm network entered
ReBeL at loss 0.0094 against the control's 0.0478.

This was the two-knob build (warm 4.0/solve, ReBeL 0.5/row). The final
one-knob code, with `train_gen_ratio = 4.0` counted per solve in both phases,
is behaviorally identical — see `runs/verify2`.
