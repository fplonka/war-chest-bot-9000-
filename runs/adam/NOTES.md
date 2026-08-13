# adam

A fresh Adam optimiser when warm-up ends: warm-up fits Greedy's values, and
`beta2 = 0.999` keeps its second moments mis-scaled for about a thousand ReBeL
steps — exactly the steps that land on the just-cleared buffer. Thirty minutes,
golden8 knobs.

Horizon 7.8% over the last quarter, `tgt_std` 0.35, 1,019 balanced solves/s.
All in the band of `base`, `seat`, `wp` and `explore`.

The Swiss ladder read **+370** against Greedy, versus `wp`'s +704. That is not a
regression: a direct 200-game GPU match of the two finals is **88-92-20, score
0.490** — the same player. Greedy loses to every net in the pool and anchors
almost nothing, and each run's ladder is over its own snapshots, so cross-run
Elo here is noise of several hundred points. The direct match is the instrument;
the ladder is for the shape of the curve inside one run (monotone here:
-105 / +175 / +229 / +327 / +370).

The change is therefore neutral at thirty minutes, and kept for the reason it
was made rather than for a measured gain.
