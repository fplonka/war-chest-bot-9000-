# v5_aged_fast16_head_l5

We were checking whether a fifth solve lane was still useful after four lanes
cleared the aged generation target. This was the same fixed post-Greedy
checkpoint and exact production stream as the four-lane run, changing only the
lane count. A per-second sampler watched both cards for the memory cost of
overlapping more large searches.

The stream completed 245,760 solves in 180.10 seconds before stopping, or
1,364.6 solves/s. Draining brought the total to 247,678 solves in 202.34
seconds. There were 823 completed games before stopping, 329 oversized
searches, 1,518 searches that hit the node limit, no exact fallback, and no
dropped work. Average GPU use was 65.6% and 64.4%; peak memory was 21,909 and
22,087 MiB. The original note reported much lower peaks because its CSV summary
compared memory fields as strings; these are the corrected numeric maxima from
the saved samples.

On the identical frozen search tape, five lanes averaged 449.0 solves/s on one
card versus 437.3 for four lanes, a 2.7% gain. The aged live endpoint improved
by 6.2%, from 1,284.9 to 1,364.6/s. The post-stop drain was longer because five
lanes had more admitted work, but startup and drain are not part of the long
steady-state target. Five lanes fit, but with only about 2 GiB of observed
headroom rather than the large margin first claimed. They became the production
choice for the next real warm-training gate, with memory monitoring required.
