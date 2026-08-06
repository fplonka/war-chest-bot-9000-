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

Two ladders, both at the gate's common settings (depth 2, 64 iters, 40
paired games per pairing, run from a worktree at the training commit):

- Turbo-vs-turbo (`runs/ladder_turbo`): **t64_turbo_s14 dominates
  t512_turbo_s15 at every snapshot** — final 1099 vs 767, s4 1051 vs 740.
  The T=512 run's better-converged targets did not make up for its data
  volume (191 epochs vs 819; the 2M buffer never fully turned over).
- Finals only (`runs/ladder_finals`, run1/run2/run3 finals + both turbo
  finals + refs): ctrl final 919, **t64_turbo_s14.final 904**, wide final
  841, t256 rerun final 793, t512_turbo_s15.final 634, greedy 226,
  random 0. Standard errors ±28-31.

Verdict for gate 3a: **no detected change** (904 vs 919, within the ±25
band). Turbo at T=64 is not visibly better than the plain run at the same
settings — but note this bundles the depth-counting fix and per-solve step
counting, so it is "the new base", not "turbo alone". The interesting
number is 3b's: turbo made T=512 affordable, and T=512 lost anyway — data
volume beats target quality, consistent with the under-training story.

Per-epoch loss settled around 0.011-0.013, similar to run 1's 0.013, on
targets that are better-converged (the turbo target's solvererr is 0.00036
vs 0.0008 at T=64). Age buckets stayed close together all run
(old ≈ new ≈ total), i.e. no visible staleness gap. tgt_std climbed slowly
(0.53 -> 0.64), pstd 0.34 -> 0.42 — the value function keeps spreading,
no collapse.

## State of the project at this point

- master has moved on (policy-head / action-features / CFR-variants work
  landed after these runs); the ladders ran from a worktree at 4aa86ca,
  the commit the runs actually used.
- Pool: s11.final (919), t64_turbo_s14.final (904), per the finals ladder.
