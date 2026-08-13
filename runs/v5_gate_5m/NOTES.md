# Five-minute v5 GPU gate

## What we were trying

This was the first full five-minute production-schedule check of the v5 two-GPU trainer after bounding each worker's outstanding searches. It used the published warm checkpoint, the documented depth-two, 64-iteration ReBeL settings, and four optimizer rows for every completed solve. The question was whether the live training loop could sustain 1,200 balanced solves per second without building training debt or running past its deadline.

## What we learned

It did not pass. The run completed 104,018 solves in 329.7 seconds: 315 balanced solves per second including the drain, with a 29.7-second overrun. The first 31 seconds were healthy at 1,386 cumulative solves per second, but throughput fell steadily as the confidence cap annealed from 0.04 to zero and games reached harder positions. It was 600 per second when the cap reached zero at 123 seconds and 370 per second just before admission stopped.

The optimizer was not the bottleneck: it trained on 415,744 rows, finished with only 328 rows of debt, and dropped no samples. The search side accumulated 1,299 node caps and 238 oversized routes. At shutdown all 4,608 actor games still in flight were censored, showing that a small number of very long live searches can occupy the bounded worker queues for much longer than ordinary early-game work. There were no exact CPU fallbacks.

## State of the project at this point

The GPU network evaluation, device-side replay expansion, continuous trainer, and bounded 128-actor worker pools were all enabled. Short early-game measurements above 1,200 solves per second remain valid, but they do not predict sustained production throughput. The checkpoint used here came from `arch_probe_published`, whose nominal Greedy warm-up was only 30 seconds. This is not the five-minute Greedy warm-up in the production recipe, so the run should not be used to decide which long-tail game dynamics deserve optimization.

## Ladder follow-up

A 60-game-per-pair round robin compared the initial and final snapshots with Greedy and Random at the same depth-two, 64-iteration search settings. Training made the model weaker: the initial snapshot beat the final snapshot 18–3 with 39 draws, a score of 0.625. Greedy scored 0.608 against the initial snapshot and 0.717 against the final one. With Greedy fixed at zero, the fitted ratings were initial −79 ± 27 Elo, final −170 ± 27, and Random −252 ± 29.

This confirms that the run was not merely slow; it was learning in the wrong direction from an inadequately warmed value network. The next meaningful check is a fresh run with the complete five-minute Greedy phase, followed by the same ladder. Tail optimization should resume only if that run learns sensible play and still produces the same sustained search bottleneck.
