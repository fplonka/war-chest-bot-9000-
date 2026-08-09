# GPU performance goal

## Vast.ai box

Instance `47157998` is a Vast.ai PyTorch container with:

- 2 x RTX 3090 (24 GiB each), CUDA 13.0, PyTorch 2.12 + cu130;
- 2 x Xeon E5-2699 v3: 36 physical cores / 72 hardware threads;
- 125 GiB RAM;
- a 16 GiB container disk and no persistent volume.

Direct SSH:

```bash
ssh -i ~/.ssh/id_ed25519_warchest_vast -p 40588 \
  root@184.144.224.246 -L 8080:localhost:8080
```

The address is dynamic; after a stop/start, confirm the IP and SSH port in the
Vast UI. The source checkout is `/workspace/warchest-engine`. Nothing on the
instance survives recycle or destroy, so copy results off the box first.

During training, GPU 0 runs the Rust CUDA solve service and GPU 1 runs PyTorch:

```bash
python train/train.py --gpu --device cuda:1 ...
```

## Performance goal

The goal is **1,200 actual ReBeL solves per second**, sustained for 30 minutes
in a real end-to-end training run with:

- random drafts;
- depth 2;
- 64 CFR iterations;
- real trained weights;
- unchanged node cap, targets, snapshot schedule and numerical correctness.

An actual solve is one freshly submitted subgame (`Data.soff`), not every game
decision and not each TurboReBeL row derived from that solve. The rate uses
total ReBeL wall time and therefore includes CPU tree construction,
serialization, host/device transfers, CUDA solving, result collection, replay
assembly, and concurrent PyTorch updates.

At the current game and snapshot ratios, 1,200 solves/s is approximately
2,400-2,800 decisions/s, 9,600 training rows/s, and 39 million fresh solves in
a nine-hour run. Correctness gates must pass before a speed result counts.

## Optimize the training run, not the benchmark

`engine/examples/gpu_gen_bench.rs` is convenient and it is not the target. The
two do not present the same work: measured on this box, the benchmark reaches
573 solves/s on both cards while a real training run reaches 164. The reason is
visible in one number. `cap`, the fraction of decisions whose subgame hits the
node cap, is 0.4% in the benchmark and 99% in the training run, which carries
85 configs per row against the benchmark's 27. Nearly every subgame the trainer
solves is as large as the cap allows, so it is a far heavier solve, and a
benchmark speedup need not move the real rate at all.

So: measure `train.py`. Every claim of progress toward the goal has to come
from a training run's own sustained rate, and any benchmark number quoted
alongside it is context, not evidence.

That means the instrumentation has to live where the work does. Build the
engine with `--features prof` and read `prof::dump_gpu()`: the resident live
set, the waiting queue, and the admission batch sizes are what say whether the
service is starved or saturated, and they are the numbers that explained the
gap so far. Where the existing timers do not answer the question, add the timer
rather than inferring from the benchmark. Nsight on the training process is
fair game too. Profiling the real run is part of the work, not a detour from
it.

`runs/gpu1200/NOTES.md` records what has been measured and what is known about
where the time goes.
