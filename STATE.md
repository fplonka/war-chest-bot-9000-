# Longrun state

- Branch: `longrun`
- Baseline: clean worktree at task start.
- Scope: trainer logging/checkpoints/resume/diagnostics, `selfplay.rs` bounded pending queries, tests, and box validation.
- Protected regions: `train.py` Buffer/train_steps/losses; `engine/src/search.rs` and `farm.rs` except the requested accounting fix.
- Required validation: `py_compile`, train tests, and the specified box resume run.
- Implemented: JSONL epoch append, snapshot-only manifest, checkpoint log removal, minute extension resume, 100k-row resume grace with debt reset, ratcheted cap value, target-cadence probe refresh, coherent logged epoch ids, 4-digit snapshots, one-load arena packing, restart wrapper, and bounded query-drop accounting.
- Protected delegated work remains untouched: Buffer/train_steps/losses, search.rs, farm.rs, and CUDA CENSUS accounting.
- Checks so far: Python compile, shell syntax, monitor sampling, `cargo check`, Python-feature `cargo check`, the new queue-drop unit test, and all queued train tests pass. Full local Rust tests hit a pre-existing 85-second selfplay assertion failure and then timed out after 600s; no diff code caused that assertion.
- Rustfmt check is blocked by pre-existing formatting drift outside this diff. The box wrapper also passes a fake-trainer restart test.
- Resume gate first phase: `go minutes=20 snapshot_every=5` ran to about 12.5 minutes, was killed with `box.sh kill`, and left `snap_0002.pt`; remote sizes were log 1,017 B, epochs 107,883 B, and snapshots 12,788,036 / 14,015,582 / 14,015,646 B.
- Resume gate passed: the wrapper resumed `snap_0002.pt` with `minutes=25`, waited for 100,000 fresh rows, then trained; first post-grace record had `steps=0`, `optimizer_debt=0.0`, and the next had 44 steps. Exit was 0. Final remote sizes: log 1,226 B, epochs 214,482 B, snapshots 12,788,036 / 14,015,582 / 14,015,646 / 14,015,710 / 14,015,774 / 14,015,838 B.
- Dashboard verification: pulled JSONL rendered 28 panels in 4 ms; log keys are only `cfg` and `snapshots`, and checkpoints contain neither `epochs` nor `replay_rows`.
- Final: commit `1bb38a9` contains the state and logged-epoch cleanup; report the artifacts, verification, protected delegated regions, and 24-hour command.
