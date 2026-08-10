# Isolated trainer profile

We were measuring one production 1,024-row optimizer step on the second RTX 3090 without self-play competing for either the CPU or GPU. The input was the frozen 20,000-row `/workspace/warchest-trainer-sample.npz` dump and the network started from `arch_probe_published/snap_00.pt`. Five warm-up steps were discarded and the next 30 were timed with a CUDA synchronization between preparation, forward, and backward/Adam.

The mean full step was 72.32 ms (78.19 ms at p95). Replay sampling took 2.45 ms, packed-row expansion plus tensor copies took 56.19 ms, the forward pass took 3.28 ms, and backward, gradient clipping, and Adam took 10.41 ms. Batches averaged 30,753 configuration rows. This confirms that the model math is already cheap in isolation and that host expansion/copy is the dominant trainer cost; the roughly 250 ms per step seen in live training is contention, not extra neural-network work.

At this point the continuous two-card trainer was correct and committed at `cb993e5`, but its one-minute smoke sustained about 900 solves/s before the deadline reserve and 738 balanced solves/s over the fixed minute. The next measurement therefore needed to separate CPU preparation contention from GPU solve/training contention before changing the architecture.
