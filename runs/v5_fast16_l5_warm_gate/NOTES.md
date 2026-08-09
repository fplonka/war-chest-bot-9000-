# v5_fast16_l5_warm_gate

We were testing the first real trainer with the half-precision network head and
five solve lanes per RTX 3090. The run started from a fresh network, spent five
minutes learning values from completed Greedy games, then ran five minutes of
the production ReBeL generator and optimizer together. This was meant to check
steady-state speed, memory safety, and whether the generator reached completed
games rather than producing only unfinished early positions.

The ReBeL phase completed 250,936 solves and trained on 1,003,520 rows. Its
whole-phase raw rate was 838.4 solves/s and its balanced generation-and-training
rate was 838.2/s. It ended with only 224 rows of optimizer debt, no overrun, no
dropped work, and no exact fallback. The first completed ReBeL games appeared
83.1 seconds after the phase began; 1,243 games completed in all. There were 644
oversized searches and 3,200 searches that hit the node limit.

The rate began near 1,900/s, crossed 1,200/s at about 94 seconds, and settled
near 885--895/s late in the run. This confirms that the slowdown is the real
shift toward later, larger searches rather than startup overhead. The result is
about 20% faster than the previous comparable five-minute warm-training gate at
699.7/s, but it is still well below the 1,200/s goal. ReBeL-phase GPU use
averaged 63.8% and 63.2%. Retained buffers reached 22,613 and 23,331 MiB, close
enough to the 24 GiB card limit that five lanes need a retirement guard before
an unattended long run.

At this point the Greedy warm-up had clearly avoided the random-network draw
trap, and the final checkpoint had seen abundant completed-game data. The run
saved the post-Greedy checkpoint as `snap_00.pt` and the five-minute trained
checkpoint as `snap_01.pt`; their ladder comparison still had to establish that
the ReBeL updates improved playing strength.
