# gpu_golden4

## What we were trying

The trainer shares one card with ten solve lanes, and in the mature half of a
golden run an optimizer step there takes about 240 ms against the 72-101 ms the
same step takes uncontended. This run tested taking lanes off the trainer's
card and giving them to the free one: twelve lanes on GPU 0 and six on GPU 1,
using the new per-device `WARCHEST_WAVE_LANES=12,6`.

## What we learned

The contention is real and the fix works on its own terms — accumulated
training time per interval fell from about 8.7 seconds to 3.28 — but it costs
far more generation than it returns. At 200,000 solves the run was at 781
solves/s against 1,035 for the balanced ten-and-ten configuration and 1,317 for
the deeper-pipeline one. It was stopped after about ten minutes rather than
spending a further twenty on a result already several hundred solves per second
behind.

Two things are worth keeping. First, the per-device lane setting is correct and
useful even though this particular split is not: a single value still applies to
every card, and the plumbing is now there for a better ratio than 2:1. Second,
the reason the trade fails is that lanes are worth more than the trainer's
latency: a card with six lanes cannot keep itself busy, and the dispatcher's
queue-length routing then sends work it cannot absorb quickly. Twelve lanes on
one card is also slightly worse than ten (949.7 against 983.5 on the aged
stream), so this configuration was poor at both ends.

## State of the project at this point

Ten lanes on both cards remains the production setting. The trainer's
contention with solve waves stays the largest unexploited item in the mature
phase; relieving it needs the trainer's own cost to come down — the device
replay ring and graph-captured step in `docs/GPU_ARCHITECTURE.md` — rather than
starving the card it runs on.
