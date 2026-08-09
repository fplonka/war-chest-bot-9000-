# gpu1200

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

**The benchmark is not measuring the training workload.** With the crash fixed,
`gpu_gen_bench` reaches 573 solves/s on the two cards. A real training run
reaches **164**. The reason is visible in one number: `cap`, the fraction of
decisions whose subgame hits the 200,000-node cap, is 0.4% in the benchmark and
99% in the training run, and the trainer carries 85 configs per row against the
benchmark's 27. Nearly every subgame the trainer solves is as large as the cap
allows. Treating the benchmark as the stop condition for the goal is therefore
wrong, and that is the single most useful thing this run established.

**Concurrency is bounded by host memory, not by the live set.** Worker count
used to be fixed at one per core, which is wrong for threads that spend 66-84%
of their time blocked on a solve; it is now `WARCHEST_GEN_WORKERS`, and raising
it 70 -> 320 moved the benchmark from 439 to 573. But a 30-minute attempt at
256 workers x 4 games was killed by the kernel: a thousand game slots, each
holding a hundred-thousand-node tree, exhausts 125 GiB. Pick the worker count
from measured resident memory per slot, not from the core count.

**Early training produces subgames the service cannot hold at all** — 130k
nodes, 280M strategy cells, 7 GiB of arena for one solve. A refusal used to
panic the worker and take the run down; the generation loop now abandons that
game instead and counts it (`Data::dropped`). Two games in a thousand, at this
stage of training. It is a real bias toward easy positions and the honest fix
— solving those on the CPU — has not been written.

## State of the project at this point

The goal is not met: 164 solves/s sustained against a target of 1,200. What is
done is that the run no longer crashes, which it always did before, so the
measurement is now possible at all. `docs/GPU_PERF_GOAL.md` carries the numbers
and what is known about where the remaining time goes: both cards report 99%
utilisation while the service's resident live set averages 45 solves against
the 256+ its own micro-benchmark holds, so the cards are busy without being
productive, and the tick's grids shrink with the live set.

This run's snapshots are here for provenance only. Thirteen minutes of ReBeL
from a fresh network teaches nothing about strength, and no ladder was run.
