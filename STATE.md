# fastfit — optimizer step cost tracks its arithmetic

Goal: an optimizer step whose cost tracks its arithmetic, not fixed overhead.
Measured baseline (task facts): 172 ms/step at 1024 rows, 111 ms/step at 256
rows; at batch 256 the optimizer takes 8.4 s of every 10 s epoch.

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

## Plan

1. Done: device-backed replay buffer; gather, mirror and one-hot actions on
   the device; per-step telemetry as device scalars read back once a call
   (no per-step syncs); host stays ~1 ms of enqueue per step.
2. In flight: tools/step_bench.py measures each net stage (expander, trunk,
   configs, join, policy, backward, Adam) and what torch.compile / CUDA
   graphs do to the forward. Decide there whether the device time needs a
   fusion pass.
3. Gate 1: steps/s at batch 256 >= 3x baseline in a short run.
4. Gate 2: 30-min default run, 200 games vs bots/sweep3_b256 (packed from
   the existing runs/sweep3_b256 at git 9c22119; arch unchanged since),
   cfr=dcfr both seats, expected ~0.50+, solves/s vs the 200.5/s reference.

Box: 2x RTX 3090 24GB; WARCHEST_BOX_DIR=/workspace/warchest-fastfit.
Reference: runs/sweep3_b256 (batch 256, 200.5 solves/s, W675/B686/D135).
