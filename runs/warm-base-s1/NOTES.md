# warm — how long should the Greedy warm-up be?

Four arms, two seeds each, every arm given the same **25 minutes of ReBeL** and
only the Greedy warm-up before it varying, so the totals differ: 30, 27, 26 and
25.5 minutes. Holding the ReBeL phase fixed rather than the total is the whole
design: with a fixed total, a longer warm-up both changes the starting network
and takes time away from ReBeL, and a losing arm cannot tell you which of the
two did it.

The ladder is the sparse graph: consecutive checkpoints inside a run for the
curve, each run's first and final against Greedy as anchors, and each
candidate final against the control final **of the same seed** at 400 games.
Random drafts, paired seats, deterministic schedule seeds. 33 players, 46
edges.

## What we learned

**Two minutes is enough.** Final Elo, and the direct paired score against the
five-minute control of the same seed, pooled over both seeds (n = 800):

| arm | warm | total | seed 1 | seed 2 | mean | score vs base | z |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 5.0 | 30.0 | 384 | 406 | **395** | — | — |
| warm2 | 2.0 | 27.0 | 404 | 368 | **386** | 0.489 | 0.6 |
| warm1 | 1.0 | 26.0 | 373 | 365 | 369 | 0.466 | 1.9 |
| warm0.5 | 0.5 | 25.5 | 323 | 267 | 295 | 0.360 | 7.9 |

Two minutes is a tie, and it is a tie while the control is given three more
minutes of wall clock and the arm generates about as many rows in less time.
One minute is a small loss the data cannot quite resolve. Half a minute costs
100 Elo and is resolved beyond any doubt. The fitted Elo and the direct
head-to-heads agree, which is the first evidence that the new sparse graph is
sound.

**The old collapse was a bug, not the warm length.** The previous sweep, at 15
ReBeL minutes, found the short arms *unstable* rather than merely worse: one
minute gave 432 and −124 on its two seeds, half a minute collapsed on both. No
arm here collapses; the four means fall in a straight line. Between the two
sweeps came the Warrior Priest information-state fixes, zero-sum values, the
exploration-sampling fix, the Adam and replay reset at the phase boundary, and
random-draft evaluation. Which of those did it is not separated, but the old
conclusion — "keep five minutes, short warm-ups are a gamble" — no longer
holds, and the production default is now two.

## Health

Every arm ran clean: effective train ratio 0.4998–0.4999 against a target of
0.5, zero dropped solves, no exact-CPU fallbacks, at most two oversize routes
in a whole run, and belief-weighted `|v0 + v1|` at most 7e-6.
Generation held 2.6k–3.6k rows/s, and the shorter warm-ups convert their saved
time into rows — 5.46M generated for warm0.5-s1 against 4.22M for base-s1.

## Read with care

The curve checkpoints get 40 games an edge and land at ±40 Elo; they are there
to shape the curve, not to be compared. Only the finals, at 1280 games and
±10–16, carry weight. And two seeds is thin: the two warm2 finals differ by 36
Elo, about the size of the effect being measured, so nothing under ~30 Elo in
this table should be believed.
