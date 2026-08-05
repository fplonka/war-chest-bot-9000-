# t256_h384_dg64_s12 — crashed at epoch 168

**Date:** 2026-08-05

## What was run

Same schedule as t64_h384_dg64_s11 (270 min, warm 5 min, snapshots every 60
min, hidden 384, dg 64) with iters 256 and seed 12.

## What happened

At t=5334 s (epoch 168, ~89 min) the training process exited with a panic in
`gen_data`:

```
thread '<unnamed>' panicked at src/selfplay.rs:425:21:
walk desync: post-draw support does not match the game belief
```

Two snapshots exist from before the crash: init (5 min) and s1 (65 min). The
run was not restarted; the crash is recorded in TODO.md with the
investigation notes. The run was not entered into the merged ladder, so it has
no rating.

For reference: t64_h384_dg64_s11 (seed 11, iters 64) ran its full 270 minutes
without this panic.
