# t512_turbo_s15 — turbo at T=512, depth 2, 4.5 hours (gate 3b)

**Date:** 2026-08-07

## What was run

The payoff test for turbo: same recipe as 3a (`t64_turbo_s14`) at iters 512
instead of 64 — 270 min, warm 5 min, snapshots every 60 min, seed 15,
depth 2, hidden 384, dg 64, cap 2M. Without turbo this would have cost ~8x
the epoch time *and* ~8x less data per epoch; with turbo the data rate is
flat in T (11 rows per solve at T=512 vs 8 at T=64, thanks to the
log-spaced thinning) and the only cost is the solve itself.

## What happened

Full 270 minutes, 191 epochs (~2.6 min/epoch), 6 snapshots, 0 panics, no
crash. Rows per epoch ~22-23k, the same as 3a's ~18-22k — the data rate
really is roughly flat in T; the epoch time is 8x (the CFR iterations
themselves).

## What was measured

Not yet laddered at the time of writing (the turbo-vs-turbo ladder and the
finals ladder run right after). Notes from the run's own log:

- Per-epoch loss ~0.013, similar to 3a's ~0.011-0.013. Training is doing
  about the same amount of work per epoch (per-solve step count) on
  better-converged targets (solvererr at T=512 is ~0.0001 vs 0.00036 at
  T=64).
- Only 191 epochs in 4.5h — the network saw ~4.3M rows, roughly a quarter
  of 3a's 17M, and the 2M buffer never fully turned over (~44% fill at
  the end). If T=512 wins the ladder it will be on target quality, not
  data volume.
- Age buckets stayed close together, as in 3a.

## State of the project at this point

- master has moved on (policy-head / action-features / CFR-variants work
  landed after these runs); the ladders ran from a worktree at 4aa86ca,
  the commit the runs actually used.
- Ladders done: turbo-vs-turbo (runs/ladder_turbo) and finals-only
  (runs/ladder_finals). Pool: s11.final (919), t64_turbo_s14.final (904).
