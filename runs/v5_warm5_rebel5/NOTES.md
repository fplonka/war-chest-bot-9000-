# Five-minute Greedy warm-up and five-minute ReBeL retry

## What we were trying

The previous speed gate started from a checkpoint that had only a 30-second Greedy warm-up and then became weaker in self-play. This run repeated the experiment from a fresh network with the production five-minute Greedy phase, immediately followed in the same process by roughly five minutes of ReBeL. Keeping both phases together preserved the optimizer state and the real buffer reset at the transition. The questions were whether the model now learned sensible play and whether the sustained two-GPU workload still missed 1,200 balanced solves per second.

## What we learned

The warm phase completed 151 batches, or 14,496 Greedy games. Its loss fell from 0.0786 to 0.0123 while the prediction spread remained substantial, and it had no node caps or drops. ReBeL started at 1,397 balanced solves per second through 20 seconds and briefly reached 1,509, but throughput again fell as the live games advanced. The final accounting was 186,391 solves and 745,472 optimizer rows over 298.4 seconds: 624.5 balanced solves per second, with only 92 rows of optimizer debt and no deadline overrun.

The trainer again kept up, so the sustained miss is on the search side. ReBeL recorded 1,499 true node-cap fallbacks, 317 oversized GPU routes, no exact CPU fallback, no dropped samples, and 4,608 unfinished games censored at the deadline. The target distribution remained live rather than collapsing: its standard deviation began around 0.24 and reached roughly 0.32–0.34 in the later full intervals.

## State of the project at this point

This is the first sustained v5 measurement with the intended Greedy initialization. It improves materially on the invalid short-warm run, but it does not clear the performance gate. The initial and final snapshots still need the post-run ladder before we can say whether the harder workload represents useful learning. Until that result exists, the correct next action is strength validation rather than more scheduler tuning.
