# rowfix

A subgame rooted mid-coin-play collects nothing. Before this, the decision
after a capped build dropped the walk was collected, and its row was packed at
a state the row encoding cannot express. Thirty minutes, golden8 knobs.

Horizon 6.1% over the last quarter, `tgt_std` 0.32, 997 balanced solves/s.
The run's own ladder: -110 / +216 / +324 / +432 / +634 / **+626** — monotone
except for the last pair, which is one snapshot apart and inside the error bar.

The comparison that counts, 200 games each with `ladder.py --finals`:

| pair | result | score |
|---|---|---|
| greedy vs rowfix.final | 0–20–0 | 0.000 |
| greedy vs adam.final | 0–20–0 | 0.000 |
| rowfix.final vs adam.final | 96–104–200 games | 0.480 |

Both finals shut Greedy out with no draws, the golden8 shape, and rate +638 and
+652 against it. The two are the same player within noise (±0.035), so the fix
costs nothing; it was made because the rows were wrong, and roughly one
decision in five hundred is a capped build.
