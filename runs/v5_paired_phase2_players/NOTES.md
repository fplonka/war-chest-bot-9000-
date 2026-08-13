# v5_paired_phase2_players

We were testing whether both players' final root-value passes could share one
readout, backpropagation, and collection schedule. After carry snapshots are
gathered, their reach buffer is no longer needed and is always at least as
large as the value buffer, so the candidate reused it for player 1's values.
That made the players independent in memory and allowed one cooperative
backpropagation launch per root instead of two, with no new allocation.

All 16 CUDA library tests passed, including the full-wave, zero-network, and
wave-composition checks. The speed result was negative. In symmetric
interleaved 20-second runs on the frozen one-card tape, the unchanged
paired-reach build measured 578.6 and 578.1 solves/s, averaging 578.4/s. The
paired-Phase-2 build measured 566.4 and 569.4 solves/s, averaging 567.9/s. The
candidate was 1.8% slower on identical eight-root jobs.

The change was reverted. Combining the players makes each cooperative kernel
live for roughly twice as long, which reduces useful overlap among the five
independent GPU lanes; that cost was larger than the saved launch and barrier
overhead. At this point paired full-reach sweeps remain active because their
shorter per-level tasks gained 1.9%, while pairing the heavier backward sweep
does not.
