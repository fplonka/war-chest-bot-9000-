# v5_aliased_25m_learning

We were trying the requested long learning check: five minutes of Greedy value
warm-up followed by 25 planned minutes of ReBeL training, with checkpoints
every six minutes. This was meant to measure the true post-anneal throughput
plateau and produce a multi-point Elo curve, using the five-lane aliased-reach
build that passed the short balanced gate.

The run did not finish. Warm-up completed normally, and checkpoints were saved
at the start of self-play, six minutes, and twelve minutes. The first ReBeL
games finished 62 seconds after the switch and 13,585 games finished before the
failure. Once the horizon payoff reached true zero at ten minutes, balanced
throughput settled near 966--1,020 solves/s with zero optimizer debt, no drops,
and no exact fallback. This establishes a clean long-run plateau below the
1,200/s target; the generator, not training, is limiting it.

At 15.9 minutes of self-play a lane attempted to grow a 1,073,741,824-element
(four-gibibyte) arena and CUDA returned out of memory. The Python exception was
logged correctly; the process then segfaulted during teardown. Peak sampled
memory was 23,545 MiB on GPU 0 and 23,267 MiB on GPU 1. Pinning four-gibibyte
jobs to one lane delayed the old six-minute OOM but did not bound memory once
the other four lanes retained their own large buffers. At this point `init`,
`s1`, and `s2` are usable partial checkpoints, but this is not a completed
25-minute run. Long runs need a stricter retained-buffer bound (or four total
lanes) before retrying; the saved checkpoints can be laddered and used as an
initial network without pretending the failed run completed.

The partial ladder nevertheless gives a clear learning curve. With 30 paired
games per matchup, the six-minute checkpoint beat the warm-only checkpoint
26--1 with 3 draws, and the twelve-minute checkpoint then beat the six-minute
checkpoint 20--8 with 2 draws. Against Greedy the sequence went from a 4--18
loss, to a 20--1 win, to a 28--0 win. Fitted ratings rose from -169 Elo at the
warm-only checkpoint to +214 at six minutes and +377 at twelve minutes, with
Greedy fixed at zero. More training was still improving playing strength
rather than merely changing the draw rate.
