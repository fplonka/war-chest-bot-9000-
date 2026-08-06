# t64_turbo_s14 — turbo at T=64, depth 2, 4.5 hours (gate 3a)

**Date:** 2026-08-07

## What was run

The first turbo run at the overnight length, so it is directly comparable to
run 1 (`t64_h384_dg64_s11`): 270 min, warm 5 min, snapshots every 60 min,
seed 14, depth 2, iters 64. Everything else identical to run 1's recipe
(hidden 384, dg 64, cap 2M, train_gen_ratio 4). Differences from run 1, all
shipped together: turbo generation (T+1 rows per solve, thinned to the ~8
log-spaced iterates + the live belief), training steps counted per solve,
depth counted in completed coin plays (tactic micro-choices ride free), the
belief-underflow fix, and the age-bucket loss in the log.

## What happened

Full 270 minutes, 819 epochs (~20 s/epoch), 6 snapshots, 0 panics, no crash.
Rows per epoch ~18-22k (vs ~4k for run 1): the buffer turns over ~5x faster.

## What was measured

Not yet laddered at the time of writing — the turbo-vs-turbo ladder and the
finals-only ladder against runs 1-3 come after 3b. Numbers below are from
the run's own log for future comparison.

- Per-epoch loss settled around 0.011-0.013, similar to run 1's 0.013, on
  targets that are better-converged (the turbo target's solvererr is
  0.00036 vs 0.0008 at T=64).
- Age buckets stayed close together all run (old ≈ new ≈ total), i.e. no
  visible staleness gap — worth watching whether that holds on longer runs.
- tgt_std climbed slowly (0.53 -> 0.64), pstd 0.34 -> 0.42 — the value
  function keeps spreading, no collapse.

## State of the project at this point

- master has the merged turbo + all plumbing (pushed to 49f6cea).
- Pool: s11.final (1033), s11.s4 (1021).
- 3b (t512_turbo_s15, same recipe at iters 512) running next.
