# fastfit — optimizer step cost tracks its arithmetic

Goal: make an optimizer step cost track its arithmetic, not fixed Python,
transfer, or synchronization overhead.

## Baseline

The task baseline is 172 ms/step at 1024 rows and 111 ms/step at batch 256.
The batch-256 optimizer used 8.4 s of every 10 s epoch. The earlier 4-minute
profile measured about 78 ms/step in warm bursts and 145 ms/step during SoG.

## Baseline profile (fastfit_prof, 4-min run, WARCHEST_TRAIN_PROFILE=1)

Per-step breakdown at batch 256, from the [profile] lines:

- Warm phase (62-step bursts, no search): sample ~6 ms, prepare ~5 ms,
  forward ~25 ms, backward ~42 ms (GPU events) -> ~78 ms/step.
- SoG phase (1-2 steps per call, GPU shared with search): sample ~14 ms,
  prepare ~14 ms, forward ~48 ms, backward ~69 ms -> ~145 ms/step.

So: the Python gather + transfers are ~10-28 ms/step, the four per-step
synchronizations serialize the host behind the device, and the forward +
backward are ~67 ms of device time (launch/latency bound, not arithmetic:
the net is a few GFLOP). Gate 1 (3x) needs the step at ~37 ms, which means
cutting the device time as well as the host overhead.

## Work completed

The replay payload rings, CUDA mirror, one-hot action encoding, and row
expansion now stay on the training device. Replay indexes and metadata stay on
the host. Policy groups are counted at ingest, so policy loss uses compact
group IDs and `index_add_` without `torch.unique`. The trainer has one batch
producer, and the dump and parity tools use it directly.

Rust and Python share one named row schema. Rows carry `source`, true-config
indices, one White-perspective outcome, creation time, and a TD(1) flag. The
buffer writes those columns in one loop; dumps expose named host columns. Query
rows are filtered by source, and calibration derives the Black outcome instead
of storing a second encoding.

Per-step training no longer reads device scalars or synchronizes. Losses return
`(loss, stats)` without an out-parameter or caller-side stats allocation. Source
IDs and delays are retained until report time. Optimizer deadlines are required,
and CUDA training uses fused Adam plus guarded default-mode `torch.compile`.

## Measurements and gates

The fused `tools/step_bench.py` run passed on the box. At batch 256 it measured
eager full step 70.66 ms and default-mode compiled full step 42.15 ms; warm and
SoG graph compile walls were 16.39 s and 26.95 s. Fused Adam did not materially
change the eager 69.39 ms backward/clip/Adam stage. The compiled result is about
2.64x against the 111 ms baseline, below the 3x / 37 ms Gate 1 target.

A 4-minute schema and mirror smoke passed at 242.4 s with 3,140 solves and
25,088 optimizer rows. A default-compile smoke passed at 241.8 s; its first
warm graph consumed 153.3 s, then query-only training ran normally. The first
profile job used the old, non-queue-aware box script and overlapped an unrelated
queued match; its result is not a gate. After rebasing onto `redesign`,
`tools/box.sh` is queue-aware and the post-rebase smoke plus post-refactor step
benchmark are queued behind other workers without touching them.

Gate 1 is a short batch-256 result at least 3x the 111 ms baseline. Gate 2 is a
30-minute default run, then 200 color-swapped games against
`/workspace/warchest-engine/bots/base_b256`; both seats use `cfr=dcfr`. The
reference is `runs/base_b256`, 319652 solves at 197.8 solves/s.

## Validation status (2026-08-26)

Local Python syntax, Rust non-GPU compile, and diff checks pass. Local CUDA
compile is unavailable because `nvcc` is not installed. The remote policy arena
indexing test passes, including 16k-row and fat-wrap cases. The pre-rebase
schema, compile, and fused-profile jobs passed; the queue-aware post-rebase
validation remains pending. No local bot, solver, or training binary has been
run.

## Remaining validation

Wait for the queue-aware post-rebase smoke and step benchmark. If the compiled
step remains above 37 ms, Gate 1 is not met; do not hide that result with another
mode or compatibility path. Run Gate 2 only if the measured path is accepted.
Never touch jobs, tickets, pid files, or run directories started by another
session.
