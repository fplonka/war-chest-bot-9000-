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
- Baseline corpus: `/workspace/warchest-onepath/onepath-roots.bin`, 64 games, retained on box.
- Baseline farmbench: 72 workers, GPUs 0,1, 60 seconds; measured window 143.5 solves/s, 25.0 calls/solve, 17.0 ms/round, 61.0 calls/round, 9,837 rows/round.
- Removed `Nets.device`, runtime backend dispatch, `Backend::Reference` outside tests, and the bot CPU opt-in.
- Removed host-only examples, the CPU throughput probe, and CPU root generation.
- Converted the host-dependent integration suites to unit suites so the oracle can be compiled only under `cfg(test)`.
- Moved 1,266 lines of host arithmetic from `search.rs` to `search/reference.rs`, compiled only for tests.
- Removed the `Nets` wrapper; solvers now receive `Arc<Net>` directly.
- Removed CPU arena opt-ins. Static generation remains for greedy replay-format tests; SoG generation uses `SolveFarm`.

## Next

1. Make the test-only reference path compile after removing runtime dispatch.
2. Move the exact oracle out of `search.rs` and delete host-only runtime state.
3. Run the box tests and matched after benchmark.
4. Run the 30-minute training and 200-game arena gate.
5. Apply the thermo-nuclear self-review and report.
