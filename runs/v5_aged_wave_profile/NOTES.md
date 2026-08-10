# v5_aged_wave_profile

We were profiling the workload that matters after a proper Greedy value warmup. This was fixed-weight generation from `v5_lane_whale_gate/snap_00.pt`, the checkpoint saved immediately after that run's five-minute Greedy phase. It used 36 builders, 128 actors per builder, 32 submitted solves per builder, both RTX 3090s, fast tensor-core GEMMs, three direct-launch lanes per card, and the tuned four-block backpropagation/two-block reach cooperative grids. No optimizer ran.

The stream completed 206,848 solves in 180.19 seconds before admission stopped, or 1,147.9 solves/s. Drain raised the total to 208,536 solves in 192.78 seconds, or 1,081.8/s. There were 249 oversized routes, 1,276 true node caps, no exact fallback, and no drop before stop. This is about 9% faster than the earlier 1,051.5/s pre-stop control, consistent with the frozen-tape gain from the new sweep grids. Average utilization was 63.5% on GPU 0 and 60.8% on GPU 1; peak memory was 13,601 and 13,395 MiB.

The opening was not representative: cumulative throughput peaked around 1,705/s at 30 seconds and declined to 1,148/s at 180 seconds. In the final 30-second window, class 0 contained 76.8% of jobs but only 16.8% of summed lane time. Classes 2, 3, and isolated class 31 contained only 6.0% of jobs yet consumed 65.5% of summed lane time. Class 3 averaged 1.5 jobs and 1,067 MiB of mutable reservation per wave; isolated jobs averaged 4,406 MiB and about 363 ms each. The expensive tail, rather than insufficient small-wave fill, is the steady-state limiter.

An uninstrumented Nsight Systems trace captured ten seconds at roughly 130 seconds into the same trajectory. Kernel time was led by `head_entry` at 22.5%, then `reach_sweep` at 16.5%, `backprop_sweep` at 13.0%, `readout` at 11.4%, `belief_sums` at 6.1%, and tensor-core GEMMs at roughly 24% combined. GPU memory-operation time was 55.0% host-to-device copies, 33.0% device-to-host copies, and 12.1% memset, but those transfers overlap across six lanes and were not the largest individual execution phase. The next kernel experiment should target the repeated 384-wide residual LayerNorm/ReLU entry path on the large jobs, while preserving the already-measured cooperative sweep caps.

Two direct follow-ups were negative and were reverted. Computing the LayerNorm second moment in the first pass, instead of revisiting the 384 register values for variance, was neutral to about 0.4% slower on interleaved tapes. Changing the register-heavy kernel from 256 to 128 threads per block initially looked slightly faster, but the longer interleaved pair averaged 394.9 versus 404.2 solves/s; 512 threads was worse again. The existing stable variance calculation and 256-thread launch therefore remain production defaults.

Folding leaf readout into the cooperative backpropagation kernel was also a clear regression. It preserved the operation order with a grid-wide barrier and removed about 27,000 launches per ten traced seconds, but constraining readout to the four-block-per-SM cooperative grid reduced the matched tape from 398.4 to 360.4 solves/s (about 9.5%). The standalone readout kernel's wider occupancy is worth more than the launch saving, so the two kernels remain separate.

Storing the rank-64 leaf and configuration readout vectors as float16 was not
worth keeping either. Compressing both made one wave-composition strategy
probability move by 0.32. Keeping the dynamic leaf vector in float32 reduced
that worst change to 0.089, but the identical frozen tape improved only from
453.6 to 454.4 solves/s, about 0.2%. The production float32 readout vectors were
restored rather than spending correctness margin and code on a noise-level
speed change.

The retained belief snapshots were a better float16 target. They are already
probabilities, and only the two spans selected by the eventual real-game exit
need to be expanded on the CPU. Storing them as float16 on the GPU and host,
then renormalizing those selected spans after expansion, changed carried
probabilities by at most 0.000342 in the fast-path oracle checks. All fast and
precise correctness suites passed. An interleaved fixed-tape comparison averaged
452.8 solves/s versus 445.5 for float32 storage, a 1.6% improvement. More
importantly, this halves the largest retained result buffer and gives the late
large searches more memory headroom.

The scheduler's one-job memory estimate still described the old all-float32
arena, so it could route a search as an isolated 4 GiB or card-exclusive 8 GiB
job even when the CUDA allocator no longer made that reservation. The estimate
now mirrors the fast-head and carried-belief float16 layouts, with a unit test
against the real CUDA arena calculation. On the opening-position tape this was
effectively neutral (457.3 versus 453.2 solves/s, about 0.9%); that tape has no
multi-GiB late searches. Its purpose is to stop unnecessary serialization in
the aged workload, which must be checked in a live trajectory.
