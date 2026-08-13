# v5_headfull16_cap0_nsys

We were profiling the production learned-checkpoint stream after compressing the
whole dynamic head path to float16. The run used true zero horizon payoff and
five lanes per RTX 3090. Nsight delayed collection until 120 seconds so its
ten-second window described aged games, then deliberately killed the diagnostic
process; this was a profile, not a completed throughput run, and no optimizer
ran.

The bottleneck moved where expected. Reach propagation used 21.9% of summed
kernel time, backpropagation 17.8%, and leaf/config readout 14.8%, or 54.5%
together. Float16 head entry used 12.3%, belief accumulation 8.3%, and the
largest head GEMM 8.4%. In the older pre-compression profile, head entry alone
was 22.5%, while reach, backpropagation, and readout were 16.5%, 13.0%, and
11.4%. The profiles used different horizon payoffs and lane counts, so their
percentages are directional rather than a timing A/B, but they clearly say the
next large opportunity is the sparse CFR sweeps/readout rather than more work
on LayerNorm.

During the capture, host-to-device copies were 74.7% of recorded CUDA memory-
operation time, versus 15.9% device-to-host and 9.4% memset. Transfers overlap
the kernels across lanes, but immutable table upload is now the obvious second
target. The run reached 138,240 solves at 120.2 seconds before collection. At
this point all 16 CUDA library tests pass, whale lane affinity prevents retained
buffer OOM, and the full-float16 head has already shown about a 10% matched-tape
gain. The next experiment should reduce hot sparse-table traffic or CFR-state
traffic and be judged on a frozen tape before another training gate.
