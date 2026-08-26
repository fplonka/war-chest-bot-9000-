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

Per-step training no longer reads device scalars or synchronizes. Profile
labels distinguish host enqueue time from CUDA event time. The peak allocator
measurement includes Net, Adam, replay, warmup, and a training-sized dummy.

## Measurements and gates

`tools/step_bench.py` is synced to the box and queued. It measures every batch
and optimizer stage, then measures default-mode `torch.compile`, its warm and
second SoG graphs, and the eager fallback path. The startup smoke is also
queued. Both wait behind the unrelated 125-minute `b256_125` run.

Gate 1 is a short batch-256 result at least 3x the 111 ms baseline. Gate 2 is a
30-minute default run, then 200 color-swapped games against
`/workspace/warchest-engine/bots/base_b256`; both seats use `cfr=dcfr`. The
reference is `runs/base_b256`, 319652 solves at 197.8 solves/s.

## Validation status (2026-08-26)

Local syntax, compile, and diff checks pass. The remote host test for policy
arena indexing passes, including 16k-row and fat-wrap cases. No local bot,
solver, or training binary has been run.
