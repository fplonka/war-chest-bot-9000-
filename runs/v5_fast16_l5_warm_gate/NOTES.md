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

The closing 30-game-per-pair ladder confirmed that the dynamics were useful.
The final checkpoint beat the post-Greedy checkpoint 13--5 with 12 draws and
placed about 122 Elo higher. The post-Greedy checkpoint lost to Greedy 7--16
with 7 draws, while the final checkpoint beat Greedy 15--7 with 8 draws. The
final checkpoint also beat Random 14--0 with 16 draws. The sample is small, but
all three comparisons move in the expected direction: the Greedy warm-up
avoided the random-network draw trap, and five minutes of ReBeL updates made the
agent measurably stronger.
