# Static lane-affinity diagnostic

## What we were trying

The valid five-minute warm checkpoint showed that rare class-2 and class-3 jobs were split across all three queues on each GPU, leaving their waves with only 5.5 and 1.5 jobs on average. This fixed-weight diagnostic assigned one lane to each common cost band so compatible jobs would batch together. It used the same checkpoint, seed, two GPUs, 36 builders, 128 actors per builder, and 32 submitted jobs per builder as the 180-second control.

## What we learned

Static affinity was immediately worse. At 30 seconds it completed 25,600 solves, or 851 per second, versus 46,080 and 1,534 per second in the control. The experiment was stopped rather than spending the full 180 seconds on a clear regression.

The opening workload is dominated by the smallest class. Giving that class only one lane per GPU removed useful parallel execution long before the rarer late classes appeared. The useful part of the hypothesis is narrower: consolidate only the sparse tail classes while continuing to route the two common classes across every available lane.

## State of the project at this point

All 14 release CUDA tests passed before this diagnostic. The fully warmed model had already passed the learning sanity check, and the unchanged fixed-weight control reached 915.5 solves per second before stop at 180 seconds. The next scheduler version should preserve common-class concurrency and apply affinity only to class 2 and above.
