# Query reservoir cleanup

## Goal

Keep harvested query roots equal to network-queried leaves, size query scratch
for the configured rates, and restore the contract test invariant.

## Progress

- Read the task scratchpad, `AGENTS.md`, and the thermo-nuclear review standard.
- Worktree was clean at start.
- Implemented the leaf-only query event view, rate validation, assertion, and
  contract-test threshold.
- Synced the dirty worktree to `/workspace/warchest-querytidy`.
- Focused box tests are queued behind existing box jobs.

## Validation

- No local bot, solver, or training binary will be run.
- Required work will use `WARCHEST_BOX_DIR=/workspace/warchest-querytidy` and
  `tools/box.sh`.
