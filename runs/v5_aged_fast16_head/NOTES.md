# v5_aged_fast16_head

We were checking whether the faster half-precision head also helped a live,
aged self-play stream. This used the same post-Greedy checkpoint, with fixed
weights and no optimizer. The command accidentally omitted
`WARCHEST_WAVE_LANES=3`, so it ran the code default of two lanes per card rather
than the production three. The code kept the residual sum before normalization
in float32, but stored the two inputs that tensor-core matrix multiplies consume
in float16.

On a frozen set of identical searches this change was 4.7% faster than the
previous code (362.8 versus 346.5 solves/s on one card), and the fast and precise
correctness suites passed. This live seed did not reproduce that gain. It
completed 147,456 solves in 180.15 seconds before stopping, or 818.5 solves/s;
draining brought the total to 149,171 solves in 230.79 seconds. There were 54
completed games before stopping, 136 oversized searches, 699 searches that hit
the node limit, no exact fallback, and no dropped work.

The live result is not a speed comparison at all because it had only four GPU
lanes instead of six. Small numerical changes would also alter sampled actions,
after which the games and search sizes diverge. The fixed-search result
establishes that the kernels are faster; a corrected six-lane live run is needed
for the production throughput check. At this point the half-precision head was
committed, but it still needed a real warm-training gate and a ladder before
being treated as a training improvement.
