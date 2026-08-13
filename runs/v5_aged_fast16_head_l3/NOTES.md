# v5_aged_fast16_head_l3

We were correcting the previous diagnostic's lane count while checking the
half-precision head on an aged fixed-weight stream. This used three lanes per
card, the post-Greedy checkpoint, 36 builders, 128 actors per builder, and 32
submitted searches per builder. It still accidentally omitted the production
wave batching variables, so it used a 49,152-row target, 64-job limit, and 0.8
ms fill wait instead of 196,608 rows, 256 jobs, and 75 ms.

The stream completed 164,864 solves in 180.08 seconds before stopping, or 915.5
solves/s. Draining brought the total to 166,449 solves in 200.10 seconds. There
were 134 completed games before stopping, 179 oversized searches, 1,008
searches that hit the node limit, no exact fallback, and no dropped work.

This established only that restoring the third lane materially helped the
small-wave configuration. It is not comparable to the earlier 1,147.9/s aged
production run because the batching policy was still different. The next run
must spell out all scheduler variables rather than relying on `stream_bench.py`,
whose defaults intentionally favor a low-latency diagnostic rather than the
trainer's throughput settings.
