# Query reservoir cleanup

## Goal

Keep harvested query roots equal to network-queried leaves, size query scratch
for the configured rates, and restore the contract test invariant.

## Progress

- Read the task scratchpad, `AGENTS.md`, and the thermo-nuclear review standard.
- Query sampling now uses live leaves only. `absorb_queries` has a debug
  assertion that rejects non-leaf query nodes.
- Query rates are validated as finite values in `[0, 1]`, so one solve cannot
  draw more than the CUDA scratch capacity of `n * b.configs` queries.
- Query slots are explicit, Algorithm R keeps the reservoir uniform, and
  replacements coalesce to one final event per slot.
- Row truth is stored at row creation, and each solve captures one creation
  timestamp.
- The contract test again requires more than four described nodes.
- The review audit found no remaining structural defect in the diff.

## Validation

- `cargo check --features python` passes locally. No local bot, solver, or
  training binary was run; local GPU tests were unavailable because `nvcc` is
  not installed.
- On the box, the serialized GPU gate passed all 11 CUDA parity tests and the
  contract test.
- The 30-minute default run completed: 330,390 solves, 2,642,944 optimizer
  rows, and 204.2 solves/s.
- The 200-game querybel match scored 94 wins, 105 losses, and 1 draw: 0.472
  for querytidy, with paired p=0.55. This is consistent with the expected
  approximately even match.
- The baseline had a non-zero interior query share, approximately one in
  twenty. The current leaf-only sampler selects 0% interior nodes, enforced
  by the assertion.
- The code diff is 125 insertions and 95 deletions across six source files;
  including this state record, the branch diff is 161 insertions and 95
  deletions. `git diff --check` passes.
- The full repository `cargo fmt --check` remains noisy from pre-existing
  formatting outside this change; no formatter churn was added.
