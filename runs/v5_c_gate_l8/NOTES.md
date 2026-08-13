# v5_c_gate_l8

## What we were trying

This is the five-minute balanced-throughput gate: five minutes of Greedy
value-network warm-up, then five minutes of real ReBeL self-play with the
trainer running beside it on the second card. It was the first end-to-end
measurement after three changes to the GPU service — a reused host staging
buffer, a much smaller wave arena, and eight solve lanes per card instead of
five. The question was simply whether the gate's 1,200 balanced solves per
second was cleared, and by how much.

## What we learned

It cleared it with room to spare: **1,404.9 balanced solves per second**,
against 1,211.1 for the best previous build. Generation and training stayed in
step — the run ended owing 1,012 optimizer rows, just inside the 1,024 the goal
allows — and there were no dropped solves, no exact CPU fallbacks, no
oversized-wave routes at all, and a 1.45-second overrun of the wall clock.

Two things are worth remembering from the counters. First, `over=0` for the
whole run: the smaller arena moved every mature search below the boundary that
used to force it onto an exclusive whale lane, so the routing machinery that
consumed much of the previous week never fired. Second, the rate through the
ReBeL phase falls steadily — about 2,100/s in the first interval, 1,500/s at
the end — because all the games start at move one together and march into the
midgame in step. The five-minute number is therefore an average over a workload
that is still fairly cheap. It is a gate, not a prediction of the thirty-minute
run, where the same games spend most of their time in the expensive middle.

Peak card memory was 20.1 and 20.3 GiB of 24. That is uncomfortably close, and
the buffer-shrink change that followed this run brought it down to about 8 GiB
by giving back lane buffers that a single huge search had grown.

## State of the project at this point

Ten CUDA lanes per card, 36 builder threads with 128 game actors each, and 32
solves in flight per builder. All 17 CUDA and library tests pass, including
both GPU-versus-CPU oracles and the exact byte-admission check. The host is now
the binding constraint on the mature workload rather than the device: the phase
timers say a solve costs about 30.7 CPU-milliseconds, of which 19.6 is building
the tree, and the builders are busy 84% of the time while the cards are busy
about 60%.
