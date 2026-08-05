# t256_h384_dg64_s12 — crashed before it measured anything

**Date:** 2026-08-05 · **Result:** none. The run died at epoch 168 (~89
minutes) with an engine panic:

```
thread '<unnamed>' panicked at src/selfplay.rs:425:21:
walk desync: post-draw support does not match the game belief
```

## What we were trying

The T question: 256 CFR iterations per decision instead of 64, everything else
identical to `t64_h384_dg64_s11` (270 min, seed 12, snapshots every 60 min).

## What happened

`gen_data` panicked mid-generation. The walk's post-draw config support (from
the subgame tree built at the walk's root) did not equal the game's belief
after a draw. `t64_h384_dg64_s11` (seed 11) played a full 270 minutes without
hitting it, so the trigger state is rare and seed-dependent — this is a latent
bug, not a T=256 failure mode.

Assessment at the time: not an easy fix. The game belief passes through a
strategy-dependent Bayes filter on the acting player's public observation
(`Belief::from_pairs`, which drops zero-weight configs), while the tree's
support is built by public-state reachability and cannot know the posterior in
advance. When a round continues with the same player drawing immediately after
their decision, the post-draw check fires before the decision-node check
(selfplay.rs:520) ever would. Both asserts are load-bearing — a wrong fix
would silently corrupt training targets — so the run was parked instead of
risked. The working hypothesis and the investigation trail live in `TODO.md`.

Two healthy snapshots (init at 5 min, s1 at 65 min) survive in this directory,
but two points rate no curve, so the run contributes nothing to the ladder.

## What it means

The T question from elo01's notes is still open. Re-run this (same config,
seed 12) after the walk-desync bug is fixed.
