# t64_h384_dg64_s11 — 270 minutes at the elo01 settings

**Date:** 2026-08-05

## What was run

Three 270-minute runs, each with a 5-minute warm start, snapshots every 60
minutes and no per-run ladder. Each changed one or two settings from elo01:

| run | changes from elo01 | seed | outcome |
|---|---|---|---|
| t64_h384_dg64_s11 | none (270 min instead of 30) | 11 | completed, 1028 epochs |
| t256_h384_dg64_s12 | iters 256 | 12 | crashed at epoch 168, see its NOTES.md |
| t64_h512_dg96_s13 | hidden 512, dg 96 | 13 | completed, 670 epochs |

Afterwards one merged ladder was run over the two completed runs: all 12
snapshots plus Greedy and Random, 100 games per pairing, 64 CFR iterations
(`train/ladder.py runs/t64_h384_dg64_s11 runs/t64_h512_dg96_s13 --games 100
--iters 64`). The crashed run's two snapshots were not entered.

## What was measured

Elo from the merged ladder, Random pinned at 0:

| snapshot | minutes trained | t64_h384_dg64_s11 | t64_h512_dg96_s13 |
|---|---|---|---|
| init | 5 | 362 | 290 |
| s1 | 65 | 876 | 819 |
| s2 | 125 | 982 | 911 |
| s3 | 185 | 1013 | 973 |
| s4 | 245 | 1052 | 979 |
| final | 270 | 1062 | 1017 |

Greedy: 157. Standard errors: ±12 for snapshots, ±19 for inits, ±21 for
Greedy, ±25 for Random. Pairing-by-pairing results: `runs/ladder_t64x512.log`.

Also recorded: run 1 trained 1028 epochs, run 3 trained 670. Per-epoch
generation in the ReBeL phase was ~15-16 s for run 1 and ~25-29 s for run 3
(48 games per epoch). elo01's ladder (8 players, 60 games per pairing) rated
this recipe 356/748/842/852/852; that ladder is a different pool, so the two
scales are not directly comparable.

## Events

- The first attempt at the merged ladder crashed on startup with
  `slot out of range`: the installed engine module predated the growing slot
  pool (fixed 8 slots). The module was rebuilt (`maturin develop --release`
  in engine/) and the ladder reran.
- Run 2 crashed at epoch 168 (~89 min) with `walk desync: post-draw support
  does not match the game belief` (src/selfplay.rs:425). Investigation notes:
  TODO.md.

## State of the project at this point

- runs/elo01: 30 min, 6 snapshots, own ladder 356-852.
- runs/t64_h384_dg64_s11: 270 min, 6 snapshots, merged ladder 362-1062.
- runs/t64_h512_dg96_s13: 270 min, 6 snapshots, merged ladder 290-1017.
- runs/t256_h384_dg64_s12: crashed at epoch 168; snapshots init and s1 exist.
- Committed as 9535286 (runs, ladder.py, notes) and 760859b (ladder log).
