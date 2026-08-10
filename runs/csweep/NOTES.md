# csweep

## What we were trying

Find out why the corrected trainer probe sat at 195 solves/s, and whether the
concurrency ceiling found earlier still applied once the weight-publication bug
was fixed. All runs: two 3090s, depth 2, 64 iterations, random drafts, seed 1,
warm weights published before ReBeL.

## What we learned

**Most of the 195 was pipeline fill.** The probe used `--rebel-games 128`,
which is fewer games than there are game slots, so it measured ramp-up. Raising
it to 3072 with the same 128 slots gave 269 solves/s.

**The old concurrency ceiling was an artefact of the stale-weights workload.**
The 30-minute attempt that the kernel killed had 1,000 game slots holding
100k-node trees. With the warm network those trees are ~8x smaller and the same
sweep peaks at 5-6 GiB of 125 GiB.

| workers x per | slots | end-to-end | generator counter |
|---|---:|---:|---:|
| 64 x 2 | 128 | 269.4 | 523.1 |
| 160 x 4 | 640 | 350.7 | 725.7 |
| 320 x 4 | 1280 | 359.6 | 700.5 |

Worker count saturates at 160 x 4. Past that the generator counter falls even
as end-to-end creeps up, which is the usual sign of oversubscription.

**The gap between the generator counter and end-to-end is the whole problem.**
Generation runs at 700-726 solves/s and the run delivers 351-381. Timing the
region that no timer covered says it is not what anyone assumed: numpy
conversion is 0.09 s and replay insertion 3.19 s, for a whole 2.13M-row batch.
The residual is 204.6 s of a 708 s ReBeL phase and it is outside the epoch
record altogether.

The remaining suspect, untested: after training, the loop starts the next
3072-game generation and only then checks the deadline, so a run ends by
throwing away a full batch of generation it has already paid for. That would
also explain why a nominal six-minute run takes twelve.

## State of the project at this point

380.8 solves/s end to end against a 1,200 goal, up from 156 at the start of the
session. None of it came from making the GPU faster: it came from filling the
pipeline, using the cores, and one exact simplification in the trainer.

The remaining factor of three does not look reachable the same way. Generation
itself tops out near 726 solves/s in this architecture, so even a perfect
overlap and a free deadline would leave a factor of 1.65 to find on the GPU
side, where `docs/GPU_ARCHITECTURE.md` measured 26,000 kernel launches a second
at 78% "busy". That is the launch-bound profile its v5 design exists to fix.
