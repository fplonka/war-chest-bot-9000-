# One solver path

## Goal

Remove the runtime host solver. Keep CUDA as the only runtime solver and keep exact host arithmetic only as a test oracle.

## Constraints

- Never run solver, bot, training, GPU, or test binaries locally.
- Run box work with `WARCHEST_BOX_DIR=/workspace/warchest-onepath tools/box.sh`.
- Required gates: box `cargo test --features gpu`; matched farmbench before/after; default 30-minute `go`; 200-game arena against `bots/sweep3_b256`, `cfr=dcfr`, both seats.
- Keep parity exact. Stop rather than weaken it.

## Progress

- Read `AGENTS.md`, the task, and the thermo-nuclear review standard.
- Located runtime host-path entry points in `search.rs`, `farm.rs`, `bot.rs`, `selfplay.rs`, and `py.rs`.
- Baseline corpus is `/workspace/onepath-roots.bin`, outside synced trees.
- Baseline farmbench: 72 workers, GPUs 0,1, 60 seconds; measured window 143.5 solves/s, 25.0 calls/solve, 17.0 ms/round, 61.0 calls/round, 9,837 rows/round.
- Removed `Nets.device`, runtime backend dispatch, `Backend::Reference` outside tests, and the bot CPU opt-in.
- Removed host-only examples, the CPU throughput probe, and CPU root generation.
- Converted the host-dependent integration suites to unit suites so the oracle can be compiled only under `cfg(test)`.
- Moved host arithmetic from `search.rs` to test-only `search/reference.rs` and `search/reference/growth.rs`; both files stay below 1,000 lines.
- Consolidated every test-oracle arena and scratch buffer under one `cfg(test)` state.
- Removed the `Nets` wrapper; solvers now receive `Arc<Net>` directly.
- Removed CPU arena opt-ins. Static generation remains for greedy replay-format tests; SoG generation uses `SolveFarm`.
- Applied the thermo-nuclear review: `advance` is now the direct CUDA state machine and the CPU call evaluator is test-only.
- Rebased onto the current FIFO `box.sh` redesign (`4ac184a`).
- Preserved query-time reservoir sampling from the new base in the single runtime path and its test oracle.
- The old `onepath-tests`, `onepath-after`, and `onepath30` jobs were killed because the broad test job ran CPU tests while holding the GPU lock.
- Read the surviving queued jobs before restarting work. They wait behind `b256_125`.
- The inherited `onepath-*` queued jobs were terminated externally without exit files. Do not alter their files or tickets.
- This session queued its own gates as `onepath-pi-gpu`, `onepath-pi-after`, `onepath-pi-cuda30`, and `onepath-pi-arena200`. After the queue harness changed, it killed only these four owned tags and restarted them through the current FIFO harness.
- The first `onepath-pi-gpu` compile found two over-gated helpers: runtime `worker_seed` was absent, and test-only `ncells` was declared in production. Both are fixed; queue a fresh full test before the remaining gates.
- Committed those fixes as `36dcb9e` and queued the replacement full test as `onepath-pi2-gpu`.
- The recorded baseline predates the merge-base at `4ac184a`, and the terminated after-run's first sample was not comparable. Queued a clean merge-base benchmark as `onepath-pi2-base`, followed by the current branch as `onepath-pi2-after`, with the same corpus and settings.
- `onepath-pi-gpu` runs the required full `cargo test --features gpu`; `onepath-pi-after` uses the baseline farmbench settings.
- `onepath-pi-cuda30` is the default 30-minute run. `onepath-pi-arena200` copies its packed bot, sets that copy to DCFR, asserts `sweep3_b256` is DCFR, and swaps seats over 200 games.
- Re-ran the thermo-nuclear review. The runtime has one direct CUDA state machine; the host evaluator, arenas, solver, replay helpers, and backend adapter compile only under `cfg(test)`.
- `onepath-pi2-base` and `onepath-pi2-after` passed. Their 40-second matched samples were 252.5 and 254.7 solves/s, respectively (+0.9%); calls/solve and round time also matched within noise.
- `onepath-pi2-gpu` was terminated without an exit file after concurrent CUDA tests failed and long host-oracle tests continued. Its captured output has no panic bodies. A serial targeted parity rerun is queued as `onepath-pi3-parity` behind the inherited long run.

## Next

1. Follow `onepath-pi3-parity`; use its uncaptured failure to fix the gate without weakening parity.
2. Queue and pass a fresh full `cargo test --features gpu`.
3. Run the default 30-minute `go`, then the 200-game two-seat DCFR arena.
4. Complete the final diff review and report the benchmark, run, arena score, and p-value.
