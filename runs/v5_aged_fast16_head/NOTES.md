# v5_aged_fast16_head

We were checking whether the faster half-precision head also helped a live,
aged self-play stream. This used the same post-Greedy checkpoint and production
generator settings as `v5_aged_wave_profile`, with fixed weights and no
optimizer. The code kept the residual sum before normalization in float32, but
stored the two inputs that tensor-core matrix multiplies consume in float16.

On a frozen set of identical searches this change was 4.7% faster than the
previous code (362.8 versus 346.5 solves/s on one card), and the fast and precise
correctness suites passed. This live seed did not reproduce that gain. It
completed 147,456 solves in 180.15 seconds before stopping, or 818.5 solves/s;
draining brought the total to 149,171 solves in 230.79 seconds. There were 54
completed games before stopping, 136 oversized searches, 699 searches that hit
the node limit, no exact fallback, and no dropped work.

The live result is not a matched speed comparison. Small numerical changes alter
sampled actions, after which the games and search sizes diverge; the earlier run
of the old code reached 1,147.9 solves/s on a different resulting trajectory.
The fixed-search result establishes that the kernels are faster, while this run
shows that a single seeded live stream is too unstable to estimate that gain.
At this point the half-precision head was committed, but it still needed a real
warm-training gate and a ladder before being treated as a training improvement.
