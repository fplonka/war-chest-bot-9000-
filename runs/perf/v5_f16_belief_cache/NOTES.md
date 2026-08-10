# v5_f16_belief_cache

We were testing float16 storage for the static per-configuration belief
embedding (`Z`). The production belief kernel reads this 64-wide vector for
every leaf/support member on every CFR iteration and already emits a float16
GEMM operand, so the candidate halved both its cache footprint and its input
traffic. Accumulation remained float32 and the precise diagnostic path retained
the float32 cache.

All 16 fast CUDA library tests passed. Two interleaved 20-second one-card tape
runs measured 580.1 and 569.7 solves/s for the candidate, averaging 574.9/s.
The matching float32-cache build measured 575.5 and 581.8, averaging 578.7/s.
The candidate was therefore about 0.7% slower, with run-to-run drift larger
than the effect.

The experiment was reverted. Like float16 current strategy and narrow local
indices, scalar conversion did not repay the smaller reads. At this point the
remaining easy half-storage targets have either been measured neutral/negative
or are too numerically central to try without a larger expected gain. Work
should return to task mapping, arithmetic removal, or genuinely vectorised
access in the reach/backprop/readout kernels.

