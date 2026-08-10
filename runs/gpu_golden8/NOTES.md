# gpu_golden8

## What we were trying

The golden run: the exact thirty-minute command in `docs/GPU_PERF_GOAL.md`,
five minutes of Greedy value warm-up and twenty-five of real ReBeL self-play
with the trainer, on both RTX 3090s. The target is 1,200 balanced solves per
second.

## What we learned

**1,315.4 balanced solves per second**, against 1,023.5 for the first complete
run of this lineage. 1,972,259 solves and 7,888,896 optimizer rows, ending 140
rows in debt against the 1,024 the goal allows, with no wall-clock overrun. Zero
dropped solves, zero exact CPU fallbacks, one oversized route and no
card-exclusive routes in twenty-five minutes.

Four things got it there, and the order they were found in matters more than
their sizes.

*Memory first.* A lane grew its device buffers to the largest wave it had ever
served and never gave them back, so one gibibyte-sized search per lane filled a
24 GiB card. That is what killed the earlier twenty-five-minute run at fifteen
minutes, and — more importantly — it is why every attempt to add lanes or
pipeline depth ended in an out-of-memory error rather than a measurement.
Shrinking the wave arena and returning oversized buffers took peak card memory
from 22.5 to about 13 GiB at no cost in throughput, and only then could anything
else be tried.

*Then the trainer.* An optimizer step on the card it shares with ten solve lanes
took about 240 ms against 72-101 ms alone, and the same Python thread is what
drains finished solves back from Rust. `--train-stream-priority -1` cuts that to
about 2 seconds of accumulated training time per reporting interval from nine,
and is worth about 14%. That flag had been tried before and recorded as making
little difference; it was measured in a state where something else was the
limit.

*Then the host.* The tree builder cost 30.7 CPU-ms per solve against this
project's own 20 ms budget, and a third of it was bytes rather than algorithm: a
node array that reallocated from 640 while a mature subgame builds 2,039 nodes,
and a `TNode` that carried a 688-byte `State` four places read and no hot loop
did. 20.4 ms now.

*And the lane.* A lane used to launch a wave, block until it finished, unpack
it, answer it, and only then start the next — so every lane went quiet between
waves, and since they all wait on the same card they went quiet together. An
Nsight trace put 94% of both cards' idle time in gaps that all ended at the
first kernel of a wave.

## State of the project at this point

Ten lanes per card, 36 builders with 128 game actors each, 32 solves in flight
per builder, jemalloc preloaded, `taskset -c 0-35`, high-priority trainer
stream. All 17 CUDA and library tests pass, including both GPU-versus-CPU
oracles, the wave-composition check and the exact byte-admission check.

Two settings that measured *better* on a generation-only stream are not
production settings, and the reason is worth keeping: twelve lanes per card and
64 solves in flight per builder both look good with no trainer and over three
minutes, and both are worse — one of them fatally — once the trainer shares the
card and the run is long enough for the retained host memory to matter. The
generation-only stream is for scheduling and memory work; `tools/v5_steady.sh`
is what predicts a golden run.
