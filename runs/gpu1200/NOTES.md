# gpu1200

> Correction, 2026-08-09: this run's `cap` field was misread. It counts games
> reaching the 256-play horizon, not subgames hitting the 200,000-node cap.
> The first ReBeL batch also ran on the freshly initialized network because the
> warm weights had not yet been published to the solve services. The correction
> below preserves the useful crash and capacity findings without treating that
> batch as the intended training baseline.

## What we were trying

Get the CUDA solve service to 1,200 fresh ReBeL solves a second in a real
training run, the target written down in `docs/GPU_PERF_GOAL.md`. Two
RTX 3090s, both running a solve service, PyTorch sharing the second card,
depth 2, 64 CFR iterations, random drafts.

## What we learned

**The service was corrupting its own memory, and that was the blocker.** Any
generation run past about 128 concurrent games died with
`CUDA_ERROR_ILLEGAL_ADDRESS`, always in a different kernel, which is what an
illegal address looks like when it is reported asynchronously. The previous
session had chased it through most of the tick without finding it.

The cause: the build scratch pool `bg` is sized for the widest thing packed per
row, which was taken to be `max(CARD_FEATS, PILE_COUNTS)` = 25 floats. But
`pile_pe` writes its own block starting at `rows * NTYPE * DE`, and `DE` is 32.
Once an admission batch carried more than about 200,000 rows, that base ran
past the end of the pool and the kernel wrote a batch's worth of card
embeddings into whatever the driver had placed after it — usually the table
slab. A live solve would then read a float where a node index should be, and
die somewhere unrelated a few hundred milliseconds later. The fix is one line:
`DE` belongs in that maximum.

Finding it needed three things worth keeping:

- checking, under `WARCHEST_GPU_ASSERTS`, that a solve's uploaded table blob is
  still byte-identical when the solve finishes. That is what turned "some
  kernel read nonsense" into "these exact bytes changed under a live solve";
- a pair of zeroed pads either side of the table slab. Nothing writes to them,
  so what lands in them and where says which neighbour overran and how far;
- the sizes. The dirty regions were always an exact multiple of
  `NTYPE * DE` floats, for 20 to 32 solves — a batch's worth of exactly one
  kernel's output block. That is what named `pile_pe`.

`compute-sanitizer` was tried repeatedly and is not usable here: it does not
finish a four-game run in fifteen minutes.

**The benchmark is not the training target, but the original explanation of
the gap was wrong.** With the crash fixed, `gpu_gen_bench` reached 573 solves/s
on two cards while this run's first generation call reported 164. `cap=0.99`
means that 99% of games reached the 256-play horizon. It says nothing about the
tree node cap. The logged `configs=85` is configs per game decision, not per
training row.

The solve services were created before warm training and were not given the
warm snapshot at the phase transition. This whole 1,024-game batch used the
freshly initialized network, producing near-uniform play, broad beliefs,
almost all horizon games, and target standard deviation 0.0175. An exact
256-game replay measured 130 solves/s on those initial weights and 297 solves/s
on the intended warm snapshot. A later corrected trainer probe produced
healthy target spread and 156 solves/s end to end. The benchmark still cannot
be the stop condition, but "nearly every subgame hit 200,000 nodes" was never a
measurement.

**Concurrency is bounded by host memory, not by the live set.** Worker count
used to be fixed at one per core, which is wrong for threads that spend 66-84%
of their time blocked on a solve; it is now `WARCHEST_GEN_WORKERS`, and raising
it 70 -> 320 moved the benchmark from 439 to 573. But a 30-minute attempt at
256 workers x 4 games was killed by the kernel: a thousand game slots, each
holding a hundred-thousand-node tree, exhausts 125 GiB. Pick the worker count
from measured resident memory per slot, not from the core count.

**The stale initial-weight workload produced subgames the service could not
hold at all** — examples reached 130k nodes and 280M dense strategy cells. A
refusal used to panic the worker and take the run down; the generation loop now
abandons that game and counts it (`Data::dropped`). About two games in a
thousand were observed in this workload. That remains an unacceptable bias and
must become an exact oversize route, but it is not evidence about the true
node-cap frequency or the intended warm-checkpoint distribution.

## State of the project at this point

The goal was not met. The reported 164 solves/s covered generation through the
batch return, not the current training pass or the remaining replay/result
work, and the batch used the wrong weights. It is retained as a stale-network
capacity measurement, not the golden baseline. The useful state at this point
was that the service no longer crashed and its high-utilisation/small-grid
profile could finally be measured. `docs/GPU_PERF_GOAL.md` now carries the
corrected baseline and `docs/GPU_ARCHITECTURE.md` the proposed replacement.

This run's snapshots are here for provenance only. Thirteen minutes of ReBeL
from a fresh network teaches nothing about strength, and no ladder was run.
