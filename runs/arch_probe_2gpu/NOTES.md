# arch_probe_2gpu

## What we were trying

Measure the current CUDA path inside the real trainer rather than in
`gpu_gen_bench`.  This was a deliberately bounded probe: two RTX 3090 solve
services, PyTorch sharing GPU 1, random drafts, depth 2, 64 CFR iterations,
128 games in the generation batch, and the ordinary four training samples per
fresh solve.  The run had a 30-second greedy warm start and a four-minute
nominal budget; an outer timeout stopped it at 295 seconds.

## What we learned

The first ReBeL generation batch did not finish before the timeout. It had more
than 255 seconds after the warm phase to finish 128 games, so no solve-rate
result or CUDA-generated training rows were returned.

The later source audit found the primary cause: the solve services still held
their freshly initialized weights. Warm training updated the CPU copy, but the
first GPU upload happened only after the first ReBeL batch returned. A replay
of those initial weights took 250.5 seconds for 256 games and sent 254/256 games
to the horizon; the intended warm snapshot took 75.4 seconds. This probe is
therefore evidence for the weight-publication bug and the severity of stale
work, not evidence that ordinary 128-game warm-checkpoint batches necessarily
take four minutes.

The observed periods with one card idle still show that static round-robin
routing can expose imbalance, but they were not isolated from the bad workload
and should not be read as its main cause.

The probe also confirms that the existing trainer cannot enforce a wall-clock
budget while a generation call is in progress.  A 30-minute run can overrun by
one entire generation batch, so the golden performance harness needs a
continuous stream and interval-based accounting rather than epoch completion.

## State of the project at this point

This used commit `0aaa466` on the vast.ai box, before the warm-to-ReBeL weight
publication fix. The CUDA correctness tests passed, but they did not cover
weight-version plumbing through the trainer. No ReBeL rows were returned and
no training step used CUDA-generated data; only the warm-start snapshot was
written. The run was stopped by the external timeout, not by an engine error.
