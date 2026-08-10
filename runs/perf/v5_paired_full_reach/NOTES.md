# v5_paired_full_reach

We were testing whether the two independent full-reach sweeps used for CFR
snapshots and final root evaluation could share one cooperative CUDA launch.
The candidate processes both players' task lists at each tree level, then uses
one grid-wide barrier before advancing. It does the same arithmetic into the
same separate reach buffers, but removes 15 cooperative launches and duplicate
level schedules from a typical 64-iteration solve.

All 16 CUDA library tests passed, including the zero-network and full-wave
oracle tests. In symmetric interleaved 20-second runs on the frozen one-card
tape, the unchanged fused-root build measured 571.6 and 576.5 solves/s. The
paired-reach build measured 583.2 and 587.0 solves/s. Their averages were 574.1
and 585.1 solves/s respectively, a repeatable 1.9% gain on identical jobs.

The paired sweep was retained. At this point the fast head and half-precision
carried beliefs are active, root averaging is folded into the ordinary reach
sweep, and large four-gibibyte jobs are pinned to one lane to avoid retaining a
whale allocation on every lane. The next scheduling question is whether two
whale lanes recover mature-stream throughput while remaining inside GPU memory.
