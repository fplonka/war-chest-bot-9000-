# base30 — the first run on the centred seat bit, and the first kept dump

Thirty minutes, five of them greedy warm-up, default configuration, dump on.
Commit `9936e7b` plus the CUDA kernel fix. This run exists to do three things at
once: check that centring the seat bit did not break learning, produce a replay
dump (no run had ever successfully kept one), and give a reference checkpoint,
because every checkpoint from before the seat change is no longer comparable.

## What we were trying

`runs/solvererr_g8` found the value network is not zero-sum, and an adversarial
review traced the constant part of it to the seat scalar being fed to the
network as a raw `0`/`1` instead of a centred `-0.5`/`+0.5`. Centring it changes
what every weight means, so this is the first run of a network that is not
comparable to anything before it.

## What we learned

**The change is neutral on the learning dynamics, which is what we wanted.**
Target statistics land on `base4h`'s curve almost exactly:

| | base4h at 10 min | base30 at 10 min |
|---|---|---|
| target mean / spread | -0.006 / 0.250 | -0.006 / 0.250 |

**The dump exists.** 2,000,000 rows and 77,357,998 configs. Every previous
attempt was destroyed by the drain-abort bug (`ad0ff20`).

**Throughput recovered most of what `base4h` had lost.** Cumulative solves at
matched ReBeL time, against `runs/gpu_golden8`:

| ReBeL min | golden8 | base4h | base30 |
|---|---:|---:|---:|
| 5 | 463,872 | 0.85 | 0.91 |
| 10 | 824,320 | 0.77 | 0.89 |
| 15 | 1,225,728 | 0.75 | 0.84 |
| 25 | 1,972,259 | 0.78 | **0.83** |

`base4h` sat 25% below golden8; `base30` is 17% below. What changed between them
was the machine, not the source: a stray script had been running on the box for
25 hours, along with tensorboard and syncthing, and neither jemalloc nor the
tuned wave settings were in use. All of that is fixed.

The remaining 17% is unattributed and is not worth more time. It is also not a
clean comparison: the centred seat bit changes the network, so `base30` plays a
different game than golden8, with different game lengths and tree sizes.

## The throughput regression that was not one

Worth recording because it cost a lot of time and two of my conclusions were
wrong before the measurement settled it.

The golden8-era commit `0192e4a` was rebuilt on the clean box and run beside
HEAD, nine minutes each, one minute of warm-up, same jemalloc, same settings.
Cumulative solves:

| ReBeL min | old (`0192e4a`) | new (HEAD) | new / old |
|---|---:|---:|---:|
| 2 | 119,808 | 122,880 | 1.03 |
| 4 | 169,984 | 183,296 | 1.08 |
| 6 | 236,544 | 290,816 | 1.23 |
| 7 | 292,864 | 370,688 | **1.27** |

HEAD is faster. There is no code regression, and the search for one in
`net.rs`, `search.rs`, `selfplay.rs` and `py.rs` found nothing because there was
nothing to find.

Two mistakes on the way, both from reading the wrong statistic. First, I
compared band-averaged instantaneous rates and concluded golden8 "does not
decay"; cumulative solves at matched age shows it plainly does. Second, I
compared golden8's 25 minutes against `base4h`'s 4-hour average and called the
difference a late-run effect; at matched age the deficit is there from minute 10.
**Cumulative solves at matched time is the honest measure. Instantaneous rates
in bands are too noisy to compare runs with.**

One real defect was found and fixed on the way (`87b108a`): `Buffer.add`
concatenated solve offsets onto an array that was never trimmed, so a 4-hour run
carried ~9M entries and copied all of them on each of ~8,600 adds. It is
quadratic in run length and sits on the thread that drains the generator. It was
present in golden8's code too, and `add_s` does not grow measurably over
`base4h`, so it was not the cause of anything observed — but a hero run is where
it would start to matter.

## State of the project at this point

The seat bit is centred in all three implementations — `net.rs`,
`value_net.py` and `gpu/wave_kernels.cu`. Missing the third one broke a run
first: the trainer and the solver disagreed about what the network computes,
targets saturated at `tgt_mean +0.93`, and generation fell from 2,560 to 101
solves/s in four minutes. `gpu::tests::full_wave_oracle` is the check for that
and it had not compiled since `keep_states` was added to `Cfg`. It compiles and
passes now, as does the whole 87-test CPU suite, which could not build at all on
a machine without CUDA.
