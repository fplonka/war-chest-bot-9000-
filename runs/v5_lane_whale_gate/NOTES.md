# Lane-local whale five-minute gate

## What we were trying

The previous GPU service stopped an entire card, released all three lanes, and ran every multi-gigabyte search alone. An aged Nsight trace showed only one stream active on one card while this happened. This run tested the simpler replacement: only the lane selected for the large search releases its buffers, while the other two lanes keep serving ordinary searches. It used a fresh network, the full five-minute Greedy value warm-up, and then the exact five-minute production ReBeL workload with concurrent training.

## What we learned

The warm-up completed 143 batches, or 13,728 finished Greedy games. Loss fell from 0.0785 to 0.0137, with no caps, oversized routes, fallbacks, or drops. ReBeL opened above 1,300 balanced solves per second and reached 1,498/s through 30 seconds, but fell as games moved into larger mid- and late-game searches: 1,199/s at 60 seconds, 983/s at 100 seconds, and 742/s shortly before admission stopped.

Final accounting was 208,956 solves and 835,584 optimizer rows over about 298.6 seconds of ReBeL wall time: 699.7 balanced solves per second. Debt was 240 rows, with no overrun, no exact CPU fallbacks, and no dropped samples. There were 463 lane-local oversized routes, 2,308 true node caps, 691 finished games, and 4,608 unfinished games censored at the deadline. The target standard deviation grew from about 0.25 to 0.29 rather than collapsing.

This is a real improvement over the otherwise matching fully warmed control: 624.5 to 699.7 balanced solves per second, about 12%. It is still well short of the 1,200/s gate. During ReBeL, average GPU utilization was 73% on card 0 and 71% on card 1. Peak memory was 21,473 MiB and 18,669 MiB respectively, so lane-local overlap remained inside both 24 GiB cards even with PyTorch training on card 1.

## State of the project at this point

All 14 release CUDA tests pass. The fully warmed ladder already established that this training setup learns useful play and reaches roughly Greedy strength. Lane-local whale isolation is therefore retained: it removes 55 lines of card-wide barrier machinery, keeps exact search results, has zero capacity failures, and makes both fixed-stream generation and real balanced training materially faster. The remaining miss is the aged search workload and concurrent GPU execution, not optimizer debt or missing terminal games.
