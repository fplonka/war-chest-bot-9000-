# v5_whale1_paired_cap0_aged

This was the direct control for the preceding two-whale-lane diagnostic. It
used the same paired-reach build, seed, learned checkpoint, five total lanes,
true zero horizon payoff, and three-minute live stream, but retained all common
four-gibibyte waves on the default single lane.

The run produced 184,320 solves in 180.02 seconds, or 1,023.9 solves/s overall.
From 120.14 seconds to the stop it produced 50,176 solves, about 838/s. It
finished 730 games and reported 1,507 searches that hit the node cap. There
were no allocation failures, exact fallbacks, or dropped solves. Peak sampled
memory was 20,601 MiB on GPU 0 and 22,155 MiB on GPU 1.

One whale lane clearly beat two. The two-lane run produced 177,152 solves at
983.9/s overall and about 735/s over its final minute, while also finishing
only 577 games on nearly the same number of node-capped searches. The runs
tracked almost identically through 110 seconds and separated only as the whale
traffic became dense. Concurrent whale waves therefore contend for the GPU
more than their added lane-level overlap helps. Keep one whale lane as the
production setting; the configurable count remains useful for diagnostics.
