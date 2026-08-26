# Longrun state

- Branch: `longrun`
- Baseline: clean worktree at task start.
- Scope: trainer logging/checkpoints/resume/diagnostics, `selfplay.rs` bounded pending queries, tests, and box validation.
- Protected regions: `train.py` Buffer/train_steps/losses; `engine/src/search.rs` and `farm.rs` except the requested accounting fix.
- Required validation: `py_compile`, train tests, and the specified box resume run.
- Implemented: JSONL epoch append, snapshot-only manifest, checkpoint log removal, minute extension resume, 100k-row resume grace with debt reset, ratcheted cap value, target-cadence probe refresh, 4-digit snapshots, one-load arena packing, restart wrapper, and bounded query-drop accounting.
- Protected delegated work remains untouched: Buffer/train_steps/losses, search.rs, farm.rs, and CUDA CENSUS accounting.
- Checks so far: Python compile, shell syntax, monitor sampling, `cargo check`, Python-feature `cargo check`, and the new queue-drop unit test pass. Full local Rust tests hit a pre-existing 85-second selfplay assertion failure and then timed out after 600s; no diff code caused that assertion.
- Rustfmt check is blocked by pre-existing formatting drift outside this diff. The box wrapper also passes a fake-trainer restart test.
- Next: run train tests on the box, then run the required box resume experiment and record artifacts.
