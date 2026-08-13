# zsloss

`zero_sum_w=1.0`: the squared zero-sum residual added to the value loss. The
search is untouched — no projection, no CUDA change.

It does what it says. `probe_zs` falls 0.083 → 0.012 inside the warm phase and
reaches **0.005** by the end of ReBeL, against the control's 0.055 → 0.058: a
ten-fold reduction, at 1,081 solves/s against the control's 1,081, with the
value spread intact.

It also costs strength. Against the pool, 20 games a pair: 0.375 vs `rowfix`,
0.325 vs `adam`, 0.450 vs `wp`, 0.400 vs `seat` over 40 — about 0.39 over 100
games, ~2 standard errors below even, and the same deficit the search-time
projection paid (0.390 over 400 games). Two independent ways of removing the
residual, both costing roughly 78 Elo.

One prediction failed: the horizon fraction did not fall (0.073 against the
control's 0.069). The mechanism where a positive common mode makes continuing
look better than winning is dramatic at δ=1 (collected rows +44%) and
negligible at δ=0.05.

What this does not test: a *reparameterisation*. Both interventions so far add
a force — one to the search, one to the objective. Removing the direction from
the model class adds neither.
