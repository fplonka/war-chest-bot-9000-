# v5_headfull16_l5_warm_gate

We were checking that the faster full-float16 head still learns in the real
pipeline. The run used five minutes of Greedy value-network warm-up followed
by a planned five minutes of ReBeL self-play, with the same two-GPU, five-lane
settings as the previous learning gate. The intended final check was a ladder
between the initial and trained snapshots and the fixed Greedy and Random
players.

The Greedy phase completed normally. Its held-out loss fell from about 0.017
to 0.012, prediction spread stayed nonzero, and no work was capped or dropped.
After switching to ReBeL, the run reached 133,120 solves by 403.5 seconds, a
cumulative 1,287 solves/s over the first 103.4 seconds of self-play. At the
same point the previous head implementation had reached about 121,000 solves
at 1,166 solves/s, so the roughly 10% fixed-workload head speedup carried into
training up to the failure.

This was not a completed learning gate. At 404 seconds a worker requested a
1,073,741,824-element wave arena and CUDA allocation failed. Peak observed
memory was 23,065 MiB on GPU 0 and 23,765 MiB on GPU 1, leaving essentially no
safe headroom on the 24 GiB cards. Only the post-warm-up `snap_00.pt` exists,
so there is no trained checkpoint to ladder. At this point the float16 head
passes the fast and precise correctness suites and improves aged generation
by about 10%, but five concurrent lanes need a stricter arena/admission limit
before another real training gate is trustworthy.
