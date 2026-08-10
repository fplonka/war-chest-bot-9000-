# v5_whale2_paired_cap0_aged

We were testing whether allowing common four-gibibyte waves to use two retained
lanes would undo the mature-stream slowdown caused by pinning every such wave to
one lane. The lane count was made configurable, with the proven one-lane route
remaining the default. This run used two whale lanes, five total lanes, the
learned warm-gate checkpoint, true zero payoff at the search horizon, and a
three-minute live stream.

The run completed without an allocation failure, fallback, or dropped solve.
It produced 177,152 solves in 180.05 seconds, or 983.9 solves/s overall. The
last minute produced 44,032 solves, about 735/s. It finished 577 games before
the stop and reported 1,547 searches that hit the node cap. Peak sampled memory
was 18,521 MiB on GPU 0 and 20,011 MiB on GPU 1.

Two whale lanes were therefore memory-safe in this sample, but did not produce
the hoped-for throughput gain: the earlier one-lane run's final minute was
about 751/s. That comparison also spans the new paired-reach kernel and the
live workload is not fixed, so a direct one-lane control on the same build is
still needed. At this point the paired-reach fusion is active and the model is
the checkpoint already shown by the snapshot ladder to learn and beat Greedy.
