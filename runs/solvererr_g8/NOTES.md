# solvererr_g8 — how many CFR iterations does the target actually need?

No training run, no games, no noise. `engine/examples/solvererr.rs` solves the
same 60 positions under every regret rule, reads the answer off at T = 4, 8, 16,
32, 64, 128, 256, and compares each reading to a converged reference. Leaf
evaluator was `runs/gpu_golden8/snap_04.pt`, the strongest network we had;
positions came from greedy play sampled 20–80 plies in; depth 2.

## What we were trying

Production solves at T=64 with the `linear` regret rule. Iterations are most of
the cost of a solve, so if a smaller number is good enough, the same wall clock
buys several times as much training data. The question was whether T=16 is good
enough, and whether the discounted rule (DCFR) closes the gap at low T.

## What we learned

**DCFR is uniformly better, and worth about a factor of two in iterations.**
Mean |error| in the value target against the converged reference:

| T | linear | plus | dcfr | pcfr |
|---:|---:|---:|---:|---:|
| 8 | 0.00487 | 0.00422 | 0.00338 | 0.00316 |
| 16 | 0.00180 | 0.00162 | 0.00106 | 0.00123 |
| 32 | 0.00060 | 0.00060 | 0.00031 | 0.00046 |
| 64 | 0.00025 | 0.00024 | 0.00017 | 0.00021 |

NashConv — what a best response to the solve's own average strategy would gain,
which is absolute and so actually compares the rules — tells the same story:
dcfr at T=32 (0.00049) is better than linear at T=64 (0.00057). Half the
iterations for the same solve quality.

**But the far more important number is the one in the denominator.** The
network's own held-out error is about 0.088 — the 4-hour baseline's training
loss sits at 0.0077 smooth-L1 with beta=0.5, which is an RMS of 0.088. The
target error introduced by stopping CFR at T=64 is 0.00025. The solver's bias is
**350 times smaller than the error the network makes anyway**. At T=16 with
linear it is 0.0018, still 50 times smaller. Even at T=4 — a sixteenth of
production — it is 0.0100, nine times smaller.

So the iteration count is nowhere near being what limits this system. Iterations
are buying target accuracy in a régime where the network cannot use it. That is
the opposite of what the doc comments in `config.py` and `truth.py` assume, and
it is worth saying plainly: those comments quote the bias figures correctly and
then draw the wrong conclusion from them, because they never divide by the
network's own error.

What that implies is a trade, not a free lunch. `docs/PERF.md`'s phase breakdown
(measured at 8 iterations) puts ~45% of a solve in per-iteration work and ~50%
in fixed per-solve work; scaled to T=64 the per-iteration part is ~88% of the
cost, so T=64 → 16 should be worth roughly 2–3x the solves per second, and
therefore 2–3x the data in a fixed wall clock. Since `runs/gpu_golden8` was
still climbing steeply when its 30 minutes ran out, more data is very likely
what this system wants.

**A side finding that may matter more than the iteration count.** The last table
`solvererr` prints is |v_0 + v_1|, how far the value network is from
antisymmetric. It sits at **0.0415 and does not move** across any iteration
count or any rule — it is a property of the network, not of the solve. The
reference value spread on these positions is 0.125, so the network's
antisymmetry violation is a third of the entire signal. CFR's guarantees assume
the subgame is zero-sum, and the subgame is only as zero-sum as the network is.
No loss curve shows this. It has not been chased down yet, and part of it may be
an artefact of the harness using uniform beliefs rather than the true posterior;
that needs checking before anything is concluded.

## What this does not answer

Target error is not play strength. A smaller T also makes the *played* strategy
more exploitable (NashConv 0.0042 at linear T=16 against 0.00057 at T=64) and
changes the distribution of positions self-play visits. Neither shows up here.
The honest next step is a real run at equal wall clock, and before that a
two-minute measurement of what T=16 actually buys in solves per second on this
box — the 2–3x above is extrapolated from a CPU profile, not measured on CUDA.

Two robustness checks were started and killed: the same sweep on positions from
*random* play (which keeps reserves full and so keeps the belief support near
the ~24 configs a real ReBeL decision carries — greedy play collapses it, and a
smaller support is an easier subgame) and the same sweep at depth 3. They were
costing the 4-hour baseline run about 12% of its throughput, and they are
CPU-only, so they belong alongside a GPU run rather than competing with one.
Until they land, read the table above as "on greedy-sampled depth-2 positions".

## State of the project at this point

Fresh experiment loop (`config.py`/`exp.py`/`ladder.py`/`truth.py`), first night
of it actually running on the box. Nothing in it had run on CUDA before tonight;
getting here needed four fixes, all of which would have stopped any GPU run
dead: `device="cuda"` has no index and `torch.cuda.set_device` rejects it, the
`gpu` cargo feature had not compiled since `mc_mix` was deleted, `box.sh` built
the extension without that feature at all, and `--features gpu` on the maturin
command line replaces the pyproject feature list rather than adding to it, so
the build silently fell back to cffi bindings.
