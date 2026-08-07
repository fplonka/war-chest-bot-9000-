# Work package B (CUDA solve) — handover state

Written by the agent that started B. Read this before touching the code.
The CPU solver is the oracle; nothing here replaces it.

## Where things live

- Worktree: `/Users/filip/Code/warchest-cuda` (branch `cuda-migration`), based
  on master at `5d753f9`. The main checkout (`/Users/filip/Code/warchest-engine`)
  is the pre-CUDA agent's master — do not mix the two.
- Modal CI harness: `gpu_ci.py` at the repo root (edit `FILTER` there to run
  one test). Cheap T4; cargo target + repo tarball cached in modal Volumes
  (`warchest-cargo`, `warchest-src`).

## What is DONE and verified (CPU-only, on the laptop)

- `engine/src/serialize.rs`: the TREE.md-v2 solve-job byte format
  (serializer + deserializer + round-trip tests, all green).
- `train/cfr_spec.py`: the torch CFR spec (executable oracle for the phase
  math). `--check` against `engine/examples/oracle_dump.rs` output passes on
  the laptop (2 solves, random weights).
- `engine/examples/oracle_dump.rs`, `examples/dbg.rs` (debug dumps).
- The full CPU test suite passes (79 tests) after the selfplay refactor.
- `engine/src/gpu/` compiles with `--features gpu` on the laptop (no CUDA:
  the gpu tests fail at "no driver", expected).

## What is implemented but NOT yet verified (needs CUDA)

- `engine/src/gpu/{service,client}.rs` + `kernels.cu`: the live-set service,
  phase kernels, build GEMMs, round trips, weight publication, warm-start
  stub (service's `warm_start` is a no-op — A4's policy path is not ported).
- `selfplay.rs`: `Game` state machine + `run_games_gpu` (two games/worker).
- `py.rs` + `train/train.py`: `gpu_start/gpu_set_weights/gpu_gen_data`,
  `--gpu` flag.
- `engine/src/gpu/tests.rs`: the oracle tests (phase by phase in B3's order,
  full-solve trip1/trip2, batch invariance, zero-network determinism).

## The current blocker (the only known bug)

On the T4, the `assemble_kernel` launch segfaults inside libcuda's
`cuLaunchKernel` (gdb backtrace: `Service::build_trunk` → `b2.launch(cfg2)`
at service.rs ~1321). Sequence that reproduces:
`cargo test --features gpu --lib phase_oracle` in the modal container.

Facts learned while chasing it:
- NVRTC mangles kernel names unless the kernels are `extern "C"` (fixed).
- The packed scratch (xb/h/u/h0p) must be sized BEFORE `launch_belief`
  writes it (fixed).
- The cuBLAS `ldc` for the transposed row-major GEMM is the row-major `n`
  (fixed).
- `tview` (byte-offset typed views into the table blob) uses CudaView
  transmute which preserves the offset (verified in cudarc 0.17.8 source).
- The first kernel launch (pile_pe) succeeds; the assemble launch crashes
  synchronously inside the driver. All 7 kernel args look valid. The
  hypothesis that the CudaView args are the problem is NOT yet tested —
  next step: pass owned CudaSlices for xpub/e instead of views, or launch
  assemble with a trivial kernel to isolate.
- gdb is installed in the modal image; `gpu_ci.py` reruns under gdb on
  failure automatically (look for "GDB OUTPUT" in the log).

## Key API/design notes for the next agent

- cudarc 0.17.8 (not 0.19): dynamic loading, `CudaSlice` (not CudaBuffer),
  `clone_dtod` is device->device, host->device is `alloc`+`memcpy_htod`,
  `device_ptr(&stream)` returns raw pointers, `transmute` is unsafe and
  checks lengths, kernels need `extern "C"`.
- The Rust `SolveDesc`/`WeightsDev` repr(C) structs must stay field-ordered
  with kernels.cu.
- The doc's phase-5/6 ordering in the tick differs from the Rust solver
  (RM, then propagate, then AVG) — follow the Rust code, not the table.
- `Solver` now owns its `Ctx` (Copy); `rm_block`/`avg_block` extracted from
  `step()` for the phase tests; `Back` is pub.
- The GPU tests take ~15 min on modal (root collection dominates). The
  serialized run is ~800 s; use `FILTER` in gpu_ci.py to run one test.
- Warm start (A4) is NOT implemented on the GPU; the service's `warm_start`
  is a stub. The tick kernels don't change for it; only the build/policy
  path is missing. The pre-CUDA agent's A4 decision may not even need it.

## Pre-CUDA agent status (as of handover)

`pre-cuda` (sibling session) is still running its offline comparisons
(runs/pre_cuda_random). Its final summary (D0 target T, A3 head shape, A4)
has not arrived. The service reads the head shape from the weights at
runtime, so B does not depend on the A3 number.

## What is left for "B done" (in order)

1. Fix the assemble-launch segv (see above), then rerun the four gpu tests.
2. Implement GPU warm start if A4 was adopted (service::warm_start stub).
3. B4.5 weight parity (CUDA vs PyTorch): a pyo3 `gpu_forward` test hook +
   `train/` script comparing build GEMMs with value_net.py on the box.
4. B5.5 same-weights ladder (`gpu_ladder` example, ~50/50 check).
5. B5.6 benchmark (`gpu_bench` example): solves/s, tick rate, GEMM rows,
   upload bytes, worker idle time — record the numbers.
6. The doc's B5 knobs (f16 upload, tf32, second stream) only after
   correctness; the same-weights ladder must re-pass after tf32.
7. runs/NOTES.md for the verification run.
