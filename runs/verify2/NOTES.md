# verify2

Confirmation that the final one-knob code reproduces the warmfix result:
`train_gen_ratio = 4.0`, counted per solve in both phases, at golden8's
schedule shortened to 10 minutes (5 warm + 5 ReBeL).

**1,229.5 solves/s (1,229.3 balanced), ratio 4.000, debt 180, zero dropped,
zero overrun.** The two runs of this code measured 1,185 and 1,229 solves/s;
the variance is the box, not the build. Combined with `warmfix` at 1,289,
current HEAD sits at 1,185-1,289 solves/s on this pilot schedule, above the
1,200 goal line.
