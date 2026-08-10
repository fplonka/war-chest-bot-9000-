# v5_root_average_fused

We were testing whether the root strategy-sum update needed its own CUDA
kernel. Every non-root decision row is accumulated inside the cooperative reach
sweep; the root was the only exception because it has no parent task. The
candidate performs the same root update after the reach sweep's final grid
barrier and deletes the standalone kernel and host launch.

All 16 CUDA library tests passed. Two interleaved 20-second one-card tape runs
measured 580.8 and 577.9 solves/s for the fused build, averaging 579.4/s. The
unchanged build measured 582.0 and 578.2, averaging 580.1/s. The 0.1%
difference is noise: the removed launch was already hidden by queued work from
the other lanes. The aged profile had recorded 47,765 instances of this kernel
in ten seconds, so folding it still materially reduces launch count.

The fusion was retained because it preserves operation order, removes one
kernel and one launch from every CFR iteration, and is performance-neutral.
At this point reach, backpropagation, and readout remain the dominant GPU work;
removing their arithmetic or improving their task mapping matters more than
peeling off additional small launches.

