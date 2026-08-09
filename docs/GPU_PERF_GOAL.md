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

## How to measure progress

`engine/examples/gpu_gen_bench.rs` accepts `GPU_WEIGHTS`, `GPU_SEED`, and
`GPU_CAP_VALUE`; use all three when replaying a trainer interval. All-zero
weights remain the right way to prove identical CPU/GPU game trajectories for
a scheduling-only A/B. Neither mode replaces a real training run.

Short development measurements should use the deterministic early/mid/late
production tapes proposed in `docs/GPU_ARCHITECTURE.md`, stay below five
minutes where possible, and report the full work distribution. Final progress
toward 1,200 comes from `train.py` with complete wall-time accounting and
concurrent training.

At minimum, every run logs raw and balanced solves/s, all queue and phase time,
weight version/age, tree dimensions, GPU wave occupancy, trainer debt, horizon
games, true node caps, oversize routes, and drops. If those counters do not add
up to wall time, the result is diagnostic rather than a target measurement.
