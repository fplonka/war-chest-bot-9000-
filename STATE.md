# Longrun state

- Branch: `longrun`
- Baseline: clean worktree at task start.
- Scope: trainer logging/checkpoints/resume/diagnostics, `selfplay.rs` bounded pending queries, tests, and box validation.
- Protected regions: `train.py` Buffer/train_steps/losses; `engine/src/search.rs` and `farm.rs` except the requested accounting fix.
- Required validation: `py_compile`, train tests, and the specified box resume run.
- Next: inspect current implementations and run baseline checks without launching training locally.
