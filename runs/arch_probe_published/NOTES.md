# arch_probe_published

## What we were trying

Repeat the bounded two-RTX-3090 trainer probe after publishing the warm-started
network to both solve services before the first ReBeL batch.  The run used
random drafts, depth 2, 64 CFR iterations, 128 games per generation call, 64
generation workers with two games each, a 30-second warm start, and the usual
four training samples per fresh solve.  Its nominal budget was four minutes.

## What we learned

The publication fix changes the workload completely.  Three ReBeL batches
finished with target standard deviations 0.291, 0.279, and 0.249; the old stale
run produced 0.0175.  Their horizon-game fractions were 0.47, 0.48, and 0.45,
not 0.99.  The batches returned 13,045, 13,200, and 12,554 fresh solves.

The trainer's displayed cumulative rates reached 234--250 solves/s, but those
rates are sampled when generation returns and do not include the current
batch's replay handling and training.  From the ReBeL transition at 39.8 s to
the final log at 288.3 s, the run completed and trained on 38,799 solves: 156
solves/s end to end.  The three logged training passes took only 39.7 s in
total, so substantial time remains outside the current `gen_s` and `train_s`
timers.  The run also overran its four-minute budget by 48 seconds because the
epoch barrier is not interruptible.

## State of the project at this point

This used commit `0aaa466` plus the warm-to-ReBeL publication fix being
developed with the GPU architecture proposal.  It is a diagnostic run, not a
strength comparison, and no ladder was run.  The result invalidates the old
claim that the first GPU training run showed 99% node-cap hits: that field is
the 256-play game horizon, and the old first batch also used the wrong weights.
The corrected end-to-end rate is still far below the 1,200 solves/s goal.
