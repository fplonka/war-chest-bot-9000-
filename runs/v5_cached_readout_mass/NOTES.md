# v5_cached_readout_mass

We were testing whether readout needed to sum the opponent's reach vector at
every network leaf. Belief normalization immediately before it already performs
the identical warp reduction. The candidate retains that scalar in scratch:
player 0 uses the first value slot that readout is about to overwrite, and
player 1 uses snapshot-reach storage after its last useful value has been
gathered. Terminal leaves still compute their sum normally.

All 16 CUDA library tests passed, including the full-wave, zero-network, and
wave-composition checks. Across two symmetric blocks of interleaved 20-second
runs on the frozen one-card tape, the unchanged build measured 581.8, 574.7,
582.2, and 582.5 solves/s, averaging 580.3/s. The cached build measured 581.7,
587.9, 590.1, and 583.9 solves/s, averaging 585.9/s. That is a 1.0% gain on
identical jobs.

The cache was retained. It adds no allocation and preserves the exact summation
order; it only carries the existing result across the intervening head GEMMs.
At this point paired full reach and cached readout masses are active, one whale
lane remains the production route, and the heavier paired-backprop experiment
has been rejected because it reduced multi-lane overlap.
