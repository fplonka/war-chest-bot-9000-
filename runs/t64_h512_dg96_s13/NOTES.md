# t64_h512_dg96_s13 — 270 minutes at hidden 512, dg 96

**Date:** 2026-08-05

## What was run

Same schedule as t64_h384_dg64_s11 (270 min, warm 5 min, snapshots every 60
min, iters 64, depth 2) with hidden 512 (was 384), dg 96 (was 64) and seed 13.
Completed: 670 ReBeL epochs.

Rated on the merged ladder with t64_h384_dg64_s11 (100 games per pairing, 64
iters). Ratings: init 290, s1 819, s2 911, s3 973, s4 979, final 1017;
t64_h384_dg64_s11's at the same snapshot times: 362, 876, 982, 1013, 1052,
1062. Full table: runs/t64_h384_dg64_s11/NOTES.md.

`ladder.json` in this directory is a copy of the merged ladder's file, so
`plot.py` draws both runs' curves on one chart.
