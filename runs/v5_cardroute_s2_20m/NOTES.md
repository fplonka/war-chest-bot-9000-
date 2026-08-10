# v5_cardroute_s2_20m

We were testing the memory fix prompted by the failed long run. This run loaded
that run's twelve-minute checkpoint, so it did not repeat Greedy warm-up, and
trained for twenty minutes with the real zero payoff for unfinished games from
the first solve. It used five lanes per RTX 3090 and saved checkpoints every
five minutes. The main question was whether draining and trimming a card before
a contiguous four-gibibyte arena would survive the mature workload that had
previously failed after 15.9 minutes of self-play.

The run completed normally. It crossed the old failure point and finished
947,204 solves, 15,696 games, and 3,700 optimizer steps with zero exact CPU
fallbacks and zero dropped work. The optimizer finished only 16 rows behind,
so generation was the limiter. Peak sampled memory was 18,201 MiB on GPU 0 and
21,963 MiB on GPU 1, safely below the failed run's 23,545/23,267 MiB. The first
games finished after about 83 seconds, not after many minutes. Target standard
deviation grew to about 0.40, which is inconsistent with a collapsed all-draw
target.

The safe route is too expensive to keep as the performance design. Balanced
throughput was 789.3 solves/s over the full fixed interval. It was about 895/s
from minutes five to ten, 869/s from minutes ten to fifteen, and 740/s over the
last five minutes before admission stopped. There were 604 isolated oversized
searches. The current telemetry does not distinguish the new card-wide route
from the older lane-local route, so a scheduler A/B is still needed before
assigning the whole slowdown to the card barrier, but the correctness and
memory-safety question is answered.

At this point the build is the five-lane, aliased-reach implementation at
`20f57d4`, with the new four-gibibyte card-exclusive route. The run began from
the strongest checkpoint of `v5_aliased_25m_learning`; its initial, five-, ten-,
fifteen-, and twenty-minute checkpoints are saved but have not yet been rated.
The next change should preserve a hard memory bound without stopping every lane
for these searches, then compare it on identical work before another long
training run.
