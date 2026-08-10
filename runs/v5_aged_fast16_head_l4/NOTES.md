# v5_aged_fast16_head_l4

We were testing whether a fourth solve lane per RTX 3090 could use the idle GPU
time left by the expensive late-search mix without exhausting memory. This was
the same aged, fixed-weight production stream as
`v5_aged_fast16_head_prod`, changing only the lane count from three to four.
The checkpoint came from the end of a five-minute Greedy warm-up; no optimizer
ran.

The stream completed 231,424 solves in 180.12 seconds before stopping, or
1,284.9 solves/s. Draining brought the total to 233,388 solves in 190.40
seconds, or 1,225.8/s. There were 701 completed games before stopping, 316
oversized searches, 1,412 searches that hit the node limit, no exact fallback,
and no dropped work. Average GPU use was 65.7% and 64.5%; peak memory was
17,773 and 17,823 MiB. The original note reported about 9.5 GiB because its
CSV summary compared memory fields as strings; re-reading the saved samples
numerically exposed and corrected that monitoring error.

On the identical frozen search tape, four lanes averaged 436.9 solves/s on one
card versus 418.4 for three lanes, a 4.4% kernel-pipeline gain. The live aged
endpoint improved by 8.1%, from 1,188.1 to 1,284.9/s. The memory result was
below the 24 GiB card limit, but not by as much as first reported. At this point
four lanes were safe for a real training gate, with enough headroom to test one
more lane while monitoring it correctly.
