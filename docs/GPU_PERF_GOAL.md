# GPU performance goal

The target is **1,200 balanced ReBeL solves/s**, sustained in a real 30-minute
training run on the two-RTX-3090 Vast.ai box. `docs/GPU_ARCHITECTURE.md`
contains the measured replacement design and implementation gates. This page
defines the box, workload, metric, and current baseline.

## Vast.ai box

Instance `47157998` is a Vast.ai PyTorch container with:

- 2 x RTX 3090 (24 GiB each), CUDA 13.0, PyTorch 2.12 + cu130;
- 2 x Xeon E5-2699 v3: 36 physical cores / 72 hardware threads;
- 125 GiB RAM;
- a 16 GiB container disk and no persistent volume.

Direct SSH at the time of the architecture measurements:

```bash
ssh -i ~/.ssh/id_ed25519_warchest_vast -p 40588 \
  root@184.144.224.246 -L 8080:localhost:8080
```

The address is dynamic. Confirm it in the Vast UI after a stop/start. The
checkout is `/workspace/warchest-engine`; copy results off the instance before
it is recycled or destroyed. Read `/etc/vast-agents-guide.md` before operating
the box.

The current WIP starts a solve service on both cards and runs PyTorch on GPU 1:

```bash
python train/train.py --gpu --gpu-devices 0,1 --device cuda:1 ...
```

## Exact target

The timed ReBeL phase uses:

- random drafts;
- depth 2;
- 64 linear-CFR iterations;
- real, changing production-network weights;
- optimizer batch size 1,024 and four optimizer samples per fresh solve;
- unchanged replay rows, targets, mirror augmentation, snapshot iterations,
  200,000-node safety cap, and horizon-payoff schedule.

An actual solve is one freshly completed subgame represented by a `Data.soff`
entry. It is not a game decision and it is not each of the roughly eight
TurboReBeL rows derived from one solve.

The raw rate is completed solves divided by all ReBeL wall time, including
pipeline fill, CPU building, copies, CUDA CFR/inference, result handling,
replay insertion, optimizer work, weight publication, snapshots, and the final
partial interval. The process must stop on the wall-clock deadline; it may not
finish a long epoch after 30 minutes.

The trainer must also keep the fixed ratio. Define training credit as
`optimizer_rows / 4` and report:

```text
balanced solves = min(completed solves, training credit)
```

The goal requires at least 1,200 balanced solves/s and at most 1,024 owed
optimizer samples at the end--256 solves at a ratio of four. It also requires
zero games dropped because a valid job did not fit the GPU service and all
correctness gates in `docs/GPU_ARCHITECTURE.md` to pass.

Unless an explicit algorithm experiment says otherwise, the golden run is
seed 1 with the ordinary network and optimizer settings:

```bash
python train/train.py --minutes 30 --warm-minutes 5 --warm-games 96 \
  --random-draft \
  --depth 2 --iters 64 --cfr linear --warm 0 \
  --hidden 384 --head 0 --dg 64 --rank 64 --de 32 --nres 1 \
  --batch 1024 --train-gen-ratio 4 --lr 0.001 \
  --lr-decay-frac 0.33,0.67 \
  --recent-mix 0.5 --recent-frac 0.2 --policy 0 --aux 0 --mc-mix 0 \
  --explore 0.25 --temp 2 --eval-mix 0.5 \
  --cap-value 0.04 --anneal-frac 0.4 --snapshot-every 6 \
  --cap 2000000 --cfgs-per-row 48 \
  --gpu --gpu-devices 0,1 --device cuda:1 --seed 1 \
  --ladder-games 0 --out runs/gpu_golden
```

Generation concurrency and wave sizes are implementation parameters, not part
of the algorithmic workload. The solve-rate clock covers the ReBeL interval
after the five-minute warm start; the whole process still has the stated
30-minute wall-clock budget. In this command `--cap 2000000` is the replay-row
capacity; the separate solver safety cap remains 200,000 tree nodes.

## The old cap claim was wrong

Historical GPU notes described `cap_frac = 0.99` as the fraction of subgames
that hit the 200,000-node cap. That is not what the code records.
`Data.cap_hits` is incremented only when a completed game reaches
`MAX_MAIN_PLAYS = 256`; it is a game-horizon counter. A solver that hits
`Cfg::node_cap` takes the uniform-policy fallback. Before this cleanup that
event had no production counter, and GPU capacity refusals in `Data.dropped`
were not exposed to Python. The trainer now logs both separately while
retaining `cap_hits` as a compatibility alias for the horizon counter.

Use four separately named counters going forward:

- `horizon_games`;
- `solver_node_caps`;
- `oversize_routes` or exact fallbacks;
- `dropped_games`, which must remain zero.

The frozen 1,000-root depth-2 sample in
`runs/pre_cuda_random/roots.bin` had **zero** 200,000-node-cap hits. It does
have a heavy tail in cells and configs, which is why admission must be sized by
more than nodes.

## The first GPU training baseline also used the wrong weights

`runs/gpu1200` reported 164 solves/s from its first ReBeL generation batch.
The solve services were created before warm training, but the warm weights were
not uploaded at the phase transition. That entire 1,024-game batch therefore
used the freshly initialized network. It produced near-uniform play, 99%
horizon games, very broad beliefs, and target standard deviation 0.0175. The
warm snapshot was uploaded only after the batch returned.

Exact two-card replays on the same seed and horizon payoff measured:

| weights | solves/s | horizon games | serialized cells/solve | snapshot configs/solve |
|---|---:|---:|---:|---:|
| fresh seed-1 initialization | 130 | 254/256 | 137,087 | 146,642 |
| warm `gpu1200/snap_00` | 297 | 86/256 | 17,239 | 19,283 |

The stale workload is genuinely much heavier, but not because its trees hit
the node cap. It is not the intended first training workload and 164/s is not a
valid baseline for the warm checkpoint.

With the warm weights published before ReBeL,
`runs/arch_probe_published` completed 38,799 solves over 248.5 seconds of ReBeL
wall time: **156 solves/s end to end**. Its generator-side counter reached
234--250/s, target standard deviation was 0.249--0.291, and the horizon-game
fraction was 0.45--0.48. The nominal four-minute run ended at 288.3 seconds,
showing that replay/result work is missing from the current phase timers and
that the epoch barrier cannot enforce the deadline.

The historical 573 solves/s result from `gpu_gen_bench` remains useful kernel
context, not evidence that training is close to the goal. It used a later
strong checkpoint and a different work distribution.

The retired v4 benchmark was replaced by the v5 contiguous-wave tape. On
2026-08-09, commit `c40d246` reached 717.1 solves/s on one RTX 3090 and 1,438.9
solves/s on both cards, with queue fill and drain included. Exact whale routing
and live cost isolation landed afterward, so those numbers are a historical
executor milestone rather than throughput of the current heterogeneous live
scheduler.

The warmed FP32 live-stream control completed 164,864 solves before stop over
180.1 seconds, or 915.5/s, and 839.3/s including drain. Lane-local whale
isolation raised those results to 1,051.5/s before stop and 1,009.4/s including
drain. The matching correct five-minute Greedy-warm/ReBeL training gate improved
from 624.5 to 699.7 balanced solves/s, with zero drops, no exact fallbacks, 240
rows of debt, and no overrun.

## Where it stands now

The first complete thirty-minute run is `runs/gpu_golden`: **1,023.5 balanced
solves/s** over the twenty-five-minute ReBeL interval, 1,533,928 solves, 928
rows of debt, no overrun, and zero drops, exact fallbacks and oversized routes.
Peak card memory was 13.0 GiB of 24, against the 23.5 GiB at which the previous
long run died. `runs/v5_c_gate_l8` is the matching five-minute gate at
**1,404.9 balanced solves/s**, up from 1,211.1.

The standing development measurement is a 180-second aged generation stream
from a late checkpoint with the horizon payoff at zero -- 4,608 live games all
in the midgame at once, which is the heaviest workload the goal has to sustain
(`tools/v5_stream.sh`). Against it:

| build | aged solves/s |
|---|---:|
| session start | 869.8 |
| smaller wave arena | 904.2 |
| eight lanes | 966.3 |
| lane buffers shrink back | 967.0 |
| ten lanes | 983.5 |
| jemalloc, 42 builders | 1,023.8 |
| pipelined lane | 1,041.0 |
| twelve lanes, 80 solves in flight | 1,136.8 |

The last row is the point of the whole exercise: a deeper in-flight window had
measured nothing at every earlier build, because the memory it needed did not
exist and because a lane that blocks on its own wave cannot use the extra depth
anyway. It only pays once both are fixed.

## Showing that it trained

Throughput is only half the goal: a run that generates fast and learns nothing
has not helped. The golden command carries `--ladder-games 0`, so the strength
check is a separate step over the snapshots the run saved:

```bash
python train/ladder.py runs/gpu_golden8 --games 30 --random-draft --gpu \
  --refs greedy,random
```

That is a round robin between every snapshot, Greedy and Random, fitted to one
rating each. What it has to show is a curve that rises with training time and
ends above Greedy -- `runs/v5_cardroute_s2_20m` is the shape to compare against,
where ratings went +327, +371, +412, +453, +536 over twenty minutes with Greedy
pinned at zero. A flat or non-monotone curve means the throughput was bought by
breaking the data, and the run does not count.

## How to measure progress

Three measurements, in increasing order of faithfulness and cost. Use the
cheapest one that can see the effect, and never the fast one alone: three
generation-side changes worth +31% on the aged stream moved the golden run by
nothing at all, because the aged stream has no trainer.

**`tools/v5_tape.sh`** — `engine/examples/wave_tape.rs` over a frozen root
sample, 25 seconds, deterministic. It measures the wave executor with no game
loop and no trainer. Good for attributing executor cost; bad for choosing,
because it rewards changes the live system cannot use (merging the cost classes
is +14% here and a 5% regression on a real workload).

**`tools/v5_stream.sh`** — 180 seconds of live self-play from a late checkpoint
with the horizon payoff at zero, no trainer. 4,608 games all reach the midgame
together, so it is the heaviest generation-only workload. Good for scheduling
and memory work.

**`tools/v5_steady.sh`** — nine minutes of real `train.py` with the real
trainer, but no Greedy warm-up, initialised from a late checkpoint, and
`--cap-value 0`, so the expensive workload is present from about ninety seconds
in instead of after ten minutes. This is the one that predicts a golden run.
Repeats agree within about 2% at equal cumulative solves.

Read all three with `tools/run_rate.py`, over a span of *solves* rather than
wall time. Cost per solve grows as games leave the opening, so a faster build
reaches the expensive positions sooner and its wall-time average converges back
towards a slower build's: `runs/gpu_golden3` is 27% faster than `runs/gpu_golden`
at 200,000 solves and ends at the same twenty-five-minute average.

## Historical measurement notes

`engine/examples/wave_tape.rs` accepts a flat weight file, a frozen roots file,
a root count, and a duration. `WARCHEST_TAPE_DEVICES`,
`WARCHEST_TAPE_PRODUCERS`, and `WARCHEST_TAPE_QUEUE` select the feeder setup;
the wave capacity variables remain executor tuning parameters. All-zero weights
remain the right way to prove identical CPU/GPU game trajectories for a
scheduling-only A/B. Neither mode replaces a real training run.

Short development measurements should use the deterministic early/mid/late
production tapes proposed in `docs/GPU_ARCHITECTURE.md`, stay below five
minutes where possible, and report the full work distribution. Final progress
toward 1,200 comes from `train.py` with complete wall-time accounting and
concurrent training.

At minimum, every run logs raw and balanced solves/s, all queue and phase time,
weight version/age, tree dimensions, GPU wave occupancy, trainer debt, horizon
games, true node caps, oversize routes, and drops. If those counters do not add
up to wall time, the result is diagnostic rather than a target measurement.
