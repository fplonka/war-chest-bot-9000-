# gpu_golden

## What we were trying

The first complete thirty-minute golden run: five minutes of Greedy value
warm-up followed by twenty-five minutes of real ReBeL self-play with the
trainer, on both RTX 3090s, using the exact command in `docs/GPU_PERF_GOAL.md`.
It was run after the wave arena was cut down and lane buffers were made to
shrink back, and it is the first run of this length that did not die of an
out-of-memory error.

## What we learned

It completed cleanly: **1,023.5 balanced solves per second** over the
twenty-five-minute ReBeL interval, 1,533,928 solves and 6,134,784 optimizer
rows, ending 928 rows in debt against a 1,024-row allowance, with no overrun.
There were zero dropped solves, zero exact CPU fallbacks, and — for the first
time — zero oversized whale routes: the smaller arena keeps every mature search
inside an ordinary wave.

That is short of the 1,200 target. The shape of the shortfall is the useful
part. Throughput starts near 2,100/s and falls to a plateau of about 960/s
after roughly ten minutes, as games leave the opening. The twenty-five-minute
average is mostly the plateau, so the plateau is what has to move.

At the plateau, neither side of the machine is saturated: the cards average 65%
and 70% busy and the builder threads use 28% of the box's 72 hardware threads.
Peak card memory was 13.0 GiB of 24, against 23.5 GiB for the long run that
died. The system is latency-coupled rather than resource-bound — the builders
wait on the cards and the cards wait on the builders — which is why every
attempt to fix it by adding threads or lanes alone measured nothing.

## State of the project at this point

Ten lanes per card, 36 builders with 128 game actors each, 32 solves in flight
per builder, `taskset -c 0-35`. All 17 CUDA and library tests pass, including
both GPU-versus-CPU oracles and the exact byte-admission check.

The host phase timers say a mature solve costs about 30.7 CPU-milliseconds:
19.6 building the tree, 4.3 serializing the job, 3.1 packing public features,
and the rest walking and bookkeeping. The architecture note's budget for 1,200
solves per second is 20 milliseconds, so the tree builder is the standing
obstacle. What moved throughput after this run was not more of the same
tuning but breaking the latency coupling: pipelining each lane so a wave stays
on the card while the lane prepares the next one, which then made a deeper
in-flight window pay for the first time.
