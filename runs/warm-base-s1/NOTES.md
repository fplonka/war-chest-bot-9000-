# warm — how long should the Greedy warm-up be?

Four arms, two seeds each, every arm given the same **15 minutes of ReBeL** and
only the Greedy warm-up before it varying, so the totals differ: 20, 17, 16 and
15.5 minutes. Then one 17-player ladder over each run's `init` and `final`.

Holding the ReBeL phase fixed rather than the total is the whole design. With a
fixed total, a longer warm-up both changes the starting network and takes time
away from ReBeL, and a losing arm cannot tell you which of the two did it.

## What we were trying

Cut the warm-up. It is 5 minutes of every run, a quarter of a 20-minute budget,
and the training logs made it look wasteful: the warm loss falls 0.078 -> 0.018
in the first 30 seconds, sits flat until minute 3.5, and then keeps falling
while `probe_std` collapses 0.46 -> 0.375. That reads as overfitting to the
handcrafted evaluation.

## What we learned

**Keep the five minutes.** Final Elo:

| arm | warm | total | seed 1 | seed 2 |
|---|---:|---:|---:|---:|
| base | 5.0 | 20.0 | **516** | **517** |
| warm2 | 2.0 | 17.0 | 515 | 446 |
| warm1 | 1.0 | 16.0 | 432 | -124 |
| warm0.5 | 0.5 | 15.5 | -47 | -91 |

Five minutes is the only setting where both seeds agree (516, 517). Two minutes
gets one seed there and loses 70 Elo on the other. Below that the runs do not
merely get worse, they become **unstable**: `warm1` gave 432 and -124, and
`warm0.5` collapsed on both seeds. A collapse costs the whole run, so the
expected value of a short warm-up is far worse than its average.

That is the opposite of what this experiment was launched to show, and the
opposite of what the loss curves suggested. Second time in one session that the
cheap metrics and the ladder disagreed, and the ladder was right both times.

**And the warm phase does not exist to make a good player.** The `init`
snapshots -- the network at the moment ReBeL takes over:

| arm | warm | init Elo |
|---|---:|---:|
| base | 5.0 | **-208, -154** |
| warm2 | 2.0 | -65, -62 |
| warm1 | 1.0 | -57, -84 |
| warm0.5 | 0.5 | -34, **+42** |

More warm-up gives a *worse* player. The five-minute network plays about 200
Elo below Greedy; the half-minute network plays near it. The `probe_std`
collapse was real and it does cost playing strength -- it just does not matter,
because the arms with the strongest starting networks produced the weakest
finals, and the arm with the weakest starting network produced the best.

So the warm phase's job is to put the network somewhere the bootstrap can start
from, not to make it good. Anyone who "improves" the warm phase by making its
network play better is likely to make the run worse. If the mechanism is worth
chasing, the guess is that fitting `eval_static` tightly gives values that are
poorly discriminating but *consistent*, and consistency is what a bootstrap
needs to avoid running away -- a 30-second warm phase in an earlier smoke run
produced exactly that runaway, target mean drifting +0.22 -> +0.28 with the
spread pinned at 0.09. Untested.

## State of the project at this point

Run on the centred seat bit (`35f37ee`), with jemalloc and the tuned wave
settings, on a box cleared of stray processes. All eight runs finished at their
intended lengths with no aborts.

The ladder rated `init` and `final` only. A full curve over every snapshot of
every arm would have been 33 players and 108 pairings, most of them between one
arm's `s2` and another's `s3`, which no experiment asks about.

Not in this run: the zero-sum projection, which was written and verified in
torch and the CPU solver during this experiment but deliberately not deployed,
because rebuilding mid-experiment would have put different builds in different
arms.
