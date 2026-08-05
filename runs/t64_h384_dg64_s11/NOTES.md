# t64_h384_dg64_s11 — 270 minutes of the elo01 recipe: no plateau, and a bigger network loses

**Date:** 2026-08-05 · **Result:** the agent keeps improving for the full 270
minutes; elo01's 17-minute plateau was the run length, not the agent.

## What we were trying

Three runs on one merged Elo ladder, each 270 minutes with a 5-minute warm
start, snapshots every 60 minutes, and no per-run ladder (the merged ladder is
the only strength measurement). The questions:

1. Same recipe as elo01 but 4.5 hours instead of 30 minutes: does the plateau
   elo01 showed at ~852 hold, or was it the clock?
2. T=256 instead of 64: does more search per decision help? — **crashed before
   measuring anything**, see `runs/t256_h384_dg64_s12/NOTES.md`.
3. Bigger network (hidden 512, dg 96): does more capacity help?

The merged ladder rated all twelve snapshots plus Greedy and Random on one
scale at 64 CFR iterations, 100 games per pairing. `ladder.py` grew multi-run
support for this; see the note about slots at the bottom.

## What we learned

**The plateau was the clock.** Run 1's ratings (Random = 0):

| when | rating |
|---|---|
| 5 min | 362 |
| 65 min | 876 |
| 125 min | 982 |
| 185 min | 1013 |
| 245 min | 1052 |
| 270 min | 1062 |

elo01 plateaued at 852 after seventeen minutes and looked finished. Given 4.5
hours the same recipe reaches 1062 and is still moving at the end: the steps
65→125 (+106), 125→185 (+31) and 185→245 (+39) are all real against ±12
standard errors, and only the last 25 minutes (245→270, +10) are inside noise.
The thirty-minute runs that built the earlier story were measuring the agent's
warm-up, not its ceiling.

**Bigger network is worse.** Run 3 (hidden 512, dg 96) trails run 1 at every
checkpoint: 819 vs 876 at 65 min, 911 vs 982 at 125, 1017 vs 1062 at the end.
Two knobs changed at once (width and belief width), so this does not say which
one hurts — but the direction matches the project's earlier finding that the
network memorises its data and extra capacity buys nothing
(`docs/REBEL.md` §5). Run 3's warm start is also worse (290 vs 362), so the
init's quality tracks the architecture too.

**Greedy anchors at 157 here** (174 on elo01's ladder). Same ballpark; the
anchor shifts a little with the opponent pool, which is worth remembering when
comparing single numbers across ladders.

**The merged ladder works.** `train/ladder.py runs/a runs/b --games 100
--iters 64` now rates several runs on one scale: snapshots are named
`run.label`, each run's checkpoints get their own slots, Random and Greedy are
entered once, and `ladder.json` lands in the first run directory. One real
gotcha surfaced on the way: the installed engine module predated the growing
slot pool (a fixed 8 slots), so the first ladder attempt died with
`slot out of range`; rebuilding the module fixed it. If a future combined
ladder throws that error, `maturin develop --release` first.

## State of the project at this point

- elo01 (30 min, T=64): 852, flat after 17 min.
- t64_h384_dg64_s11 (270 min, T=64): 1062, still climbing.
- t64_h512_dg96_s13 (270 min, bigger net): 1017.
- t256_h384_dg64_s12: crashed at epoch 168, unmeasured — the T question from
  elo01's notes is still open, and the next T comparison should be re-attempted
  once the walk-desync bug in `TODO.md` is fixed.
