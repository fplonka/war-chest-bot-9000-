# v5_async_guard_device_profile_90s

We were measuring where time goes inside the oversized searches handled by the
asynchronous two-lane guard. Device phase timing was enabled only for waves
whose combined arena crossed the four-gibibyte route threshold. Single-job
lines are the guarded searches; a few multi-job ordinary waves also crossed
the aggregate threshold and were ignored in the summary.

The 17 guarded searches averaged 477 ms inside the device executor. Arena and
table allocation, upload, and zeroing averaged 146 ms; queuing and executing
the solve plus its result transfers averaged 274 ms; CPU result unpacking
averaged 39 ms. The remaining capture and final synchronization time was small.
Mean guarded shape was 122,056 network rows and 1.95 million legal-action
cells. The run completed at 1,035.1 pre-stop solves/s, but profiling changes
timing and this was not intended as a speed comparison.

There were no allocation failures, exact fallbacks, or dropped solves. Sampled
peak memory was 17,849 MiB on GPU 0 and 21,899 MiB on GPU 1. At this point the
largest immediate opportunity is the arena itself: these jobs request a four-
gibibyte allocation because allocator growth rounds a raw arena just over two
gibibytes to the next power of two. Measuring the largest component arrays is
the next step; getting the raw layout below two gibibytes would remove both the
guard path and much of its setup cost.
