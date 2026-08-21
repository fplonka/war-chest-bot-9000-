# Making the solve farm fast

The number is **solves/s in a real `train.py` run**, not on the bench.

## What we learned

Fit three measured budgets: a solve costs
`1.94 us x (network rows) + 9 ns x (rows summed over CFR iterations)`.
At `expand=8, iters=64` the first term is **88%**. So:

* **Tree size is the whole budget lever.** Expansions set the rows.
* **CFR iterations are nearly free.** `c = 1`, which Student of Games
  specifies and we abandoned, costs almost nothing.
* The cards run at 19% of FP32 peak. Not flops, not bytes, not the host
  (3.4 ms a round uncontended), not solves in flight (twenty already reach
  30/s). **Small dependent kernels and a barrier.**

## The design

**One solve is a state machine, not an OS thread.** `Gate::slot()` is a
thread-local, so a round waits for all thirty-six of a cohort — and at 512
rounds a solve the farm collapsed from 34 solves/s to 4. Make
`solve_on_device` into `Solver::advance(&[Reply]) -> Step`, run one worker a
core over twenty-odd solvers, one driver a card draining an outbox. Bound
solves in flight by bytes held, not threads. Delete `Gate`, `Pending`,
`Member`, `PATIENCE_*`, `round_before`, the mailboxes, and the five-variant
`Call` enum with its per-kind shards.

**Delete lanes.** A lane is a second copy of a card — stream, staging, solve
table, weights — existing only so cohorts overlap. No cohorts, no lanes, and
no twenty NVRTC compiles.

**Capture a CFR iteration as a CUDA graph.** Fifty dependent launches an
iteration, tens of microseconds each. cudarc 0.17.8 already has stream capture.

**Delete dead knobs:** `grow_every` (28% worse exploitability at two),
`half_leaves`, `mc_mix`, `eval_mix`.

**Run `expand=1` and halve the expansions.** Halving the tree costs 1.3 of
13.6 public levels, because growth is selective. Gate on `arena.py tablebase`.
