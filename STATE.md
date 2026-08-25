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

## Next

1. Record the baseline farmbench on the box.
2. Map host-only state and tests.
3. Move the exact oracle under tests and delete runtime host state, dispatch, flags, and CPU opt-in.
4. Run the required gates and self-review the diff.
