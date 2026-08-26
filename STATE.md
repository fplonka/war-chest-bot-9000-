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
- Rebased onto the FIFO `box.sh` redesign.
- Preserved query-time reservoir sampling from the new base in the single runtime path and its test oracle.
- The old `onepath-tests`, `onepath-after`, and `onepath30` jobs were killed because the broad test job ran CPU tests while holding the GPU lock.
- Read the surviving queued jobs before restarting work. They wait behind `b256_125`.
- Restored the required full `cargo test --features gpu` command in `onepath-gpu`; `onepath-after2` uses the baseline settings.
- `onepath_cuda30` is the default 30-minute run. `onepath-arena200` copies its packed bot, sets that copy and `sweep3_b256` to their recorded DCFR search, and swaps seats over 200 games.
- Re-ran the thermo-nuclear review. The runtime has one direct CUDA state machine; the host evaluator, arenas, solver, and replay helpers compile only under `cfg(test)`.

## Next

1. Follow `onepath-gpu`, `onepath-after2`, `onepath_cuda30`, and `onepath-arena200` in order; fix any failed gate without weakening parity.
2. Compare farmbench with the recorded baseline, inspect the default run, and report the arena score and p-value.
3. Complete the final diff review and report.
