# v5_aged_fast16_head_prod

We were measuring the half-precision head on the exact aged production
generator after two earlier diagnostics accidentally used `stream_bench.py`'s
small-wave defaults. This run made every scheduler setting explicit: three
lanes per card, a 196,608-row target, a 256-job limit, a 75 ms fill wait, 36
builders, 128 actors per builder, and 32 submitted searches per builder. It
used both RTX 3090s and the checkpoint saved after the five-minute Greedy phase
of `v5_lane_whale_gate`; weights stayed fixed and no optimizer ran.

The stream completed 214,016 solves in 180.14 seconds before stopping, or
1,188.1 solves/s. Draining brought the total to 215,763 solves in 190.42
seconds. There were 566 completed games before stopping, 264 oversized
searches, 1,376 searches that hit the node limit, no exact fallback, and no
dropped work.

The previous matched production run completed 1,147.9 solves/s, so the live
gain was 3.5%. That agrees with the 4.7% gain on an identical frozen search
tape closely enough to validate the change. It also leaves the aged generator
just below the 1,200/s integration target before concurrent training is added.
At this point the next low-complexity experiment was adding one more GPU lane
to use the cards' measured idle capacity while watching the large-search memory
tail.
