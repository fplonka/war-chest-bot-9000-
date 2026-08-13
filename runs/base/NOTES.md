# base

Thirty-minute golden8-defaults run after the tooling rewrite (`train.py` plus `box.sh go`, Swiss ladder, no `exp.py`). Same player knobs as `gpu_golden8`.

Horizon 6% at the end, 1,132 balanced solves/s, debt 672. Swiss Elo vs Greedy at 0: init −102, then +346 / +413 / +534 / **+617** (95% CI about ±53). Monotone through the snapshots. Healthy; the next gate is the centred seat bit.

Tooling at this point: jemalloc, pin one thread per physical core, refuse a busy box. Ladder was still 100 games × snapshots (500 games); that default is now 40.
