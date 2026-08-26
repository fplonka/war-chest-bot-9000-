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
- The session-owned stale `0-b256_125_vs_v5` job was killed and its ticket
  removed. The queue is now behind other workers' `probe_rr16` and `onepath-*`
  jobs, which remain untouched.

## Validation

- `cargo check` and `cargo check --features python` pass locally.
- The GPU check failed locally because `nvcc` is not installed; no local bot,
  solver, or training binary was run.
- Required work uses `WARCHEST_BOX_DIR=/workspace/warchest-querytidy` and
  `tools/box.sh`.
