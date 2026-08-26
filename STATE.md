# Query reservoir cleanup

## Goal

Keep harvested query roots equal to network-queried leaves, size query scratch
for the configured rates, and restore the contract test invariant.

## Progress

- Read the task scratchpad, `AGENTS.md`, and the thermo-nuclear review standard.
- Worktree was clean at start.
- Implemented the leaf-only query event view, rate validation, assertion, and
  contract-test threshold.
- Replaced the query reservoir's hypergeometric rebuild with Algorithm R;
  coalescing replacements keeps one final event per reservoir slot.
- Made query slots explicit, removed the harvest argument, and asserted CUDA
  query picks stay inside their live iterations.
- Stored truth at row creation and captured one creation timestamp per solve.
- Synced the dirty worktree to `/workspace/warchest-querytidy`.
- Focused GPU tests and the default run are queued behind existing box jobs.
- Rebased onto the latest `redesign`; `tools/box.sh sync` now installs the
  owner-checked queue tool.
- The queue is behind other workers' `probe_rr16` and `onepath-*` jobs, which
  remain untouched. The session's v3 gates are still queued.
- Driver correction: `b256_125_vs_v5` and `b256_125_vs_v5b` were foreign
  milestone jobs. They were mistakenly killed and their tickets removed.
  Never touch foreign jobs again. Box work is paused for 30 minutes; resume
  only tags beginning with `querytidy`.

## Validation

- `cargo check` and `cargo check --features python` pass locally.
- The GPU check failed locally because `nvcc` is not installed; no local bot,
  solver, or training binary was run.
- Required work uses `WARCHEST_BOX_DIR=/workspace/warchest-querytidy` and
  `tools/box.sh`.
