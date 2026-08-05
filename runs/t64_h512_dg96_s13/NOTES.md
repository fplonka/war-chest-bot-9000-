# t64_h512_dg96_s13 — the bigger network, 270 minutes

**Date:** 2026-08-05 · **Result:** more capacity loses at every checkpoint.

One change from the baseline run: hidden 512 (was 384) and dg 96 (was 64) — a
bigger value network and a wider belief embedding, both at once. Everything
else identical (T=64, depth 2, seed 13, 270 min, snapshots every 60 min).

Rated on the merged ladder with `t64_h384_dg64_s11` (see that run's NOTES.md
for the full table). The bigger net trails the baseline everywhere: 819 vs 876
at 65 min, 911 vs 982 at 125, 1017 vs 1062 at the end, and its init is worse
too (290 vs 362). Consistent with the project's standing finding that the
network memorises and capacity buys nothing (`docs/REBEL.md` §5) — the new
result is that this holds at 270 minutes, not just 30.

`ladder.json` here is a copy of the merged ladder's file, so `plot.py` shows
both runs' curves on one chart.
