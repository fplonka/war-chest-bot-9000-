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

**A side finding that is two hundred times larger than the one above.** The last
table `solvererr` prints is |v_0 + v_1|, how far the value network is from
antisymmetric. In a zero-sum game the two players' values must sum to zero. This
one sits at **0.0415 and does not move** across any iteration count or any rule.

That flatness is the diagnosis: a quantity CFR manufactured would change as CFR
converges. It does not, so the violation is in the network's leaf values and the
solve merely inherits it.

It is also not an artefact of `solvererr`'s uniform beliefs. Measured on a set
of positions solved to convergence under the *real* self-play posterior
(`truth.py`-generated, 1,334 positions, T=1024 dcfr):

| quantity | value |
|---|---:|
| mean `v_0 + v_1` | **+0.0502** |
| mean &#124;`v_0 + v_1`&#124; | 0.0567 |
| std `v_0 + v_1` | 0.0451 |
| std `v_0` (the signal) | 0.3545 |

Two components, and they matter very differently. The **mean** +0.05 is a
uniform offset, and a constant added to all of one player's values does not
change that player's best response — it is embarrassing but probably harmless to
play. The **std** 0.045 is the part that bites: it is state-dependent, so the
network hands out a bonus in some positions and a penalty in others, which
distorts the value of one state relative to another and therefore does change
play. That component is roughly 13% of the value signal, and it is **180 times
larger than the target bias from stopping CFR at T=64** that this whole sweep was
about.

Worth noting what it is not: the mirror augmentation does swap seats, but it
maps a position to a *different* position, which asks the network for
equivariance, not antisymmetry. Nothing in training has ever asked for
`v_0(s) = -v_1(s)` at the same `s`. One suspicious coincidence to chase: the
horizon's marker payoff `cap_value` is 0.04, the same size as the offset, and
golden8 annealed it to zero with only ~13 minutes of training left to unlearn
it. Not tested.

**What kind of error it is.** The violation correlates **+0.67** with
`(v_0 - v_1)/2` — how far ahead player 0 is — but **-0.05** with the *magnitude*
of that lead. A symmetric miscalibration would show the opposite pattern. What
fits is a difference in value *scale* between the two seats: the network reads
the leader's advantage on a slightly different ruler depending on which seat the
leader is in. It is also concentrated early: averaged over fifths of a dump in
play order the violation runs 0.088, 0.088, 0.074, 0.011, -0.010, fading to
nothing near terminal positions where the actual win pins the value and the
network has no room to be inconsistent. Which is to say it is largest exactly
where the search depends on the network most.

`search.rs` already knew the violation exists — `Conv::zero_sum` documents it
and correctly says it is a property of the network rather than a bug in the
solver. What was missing was its size relative to the signal, which is what
makes it worth acting on rather than noting.

**Where it comes from: nowhere, which is the point.** Measured directly as the
network's own belief-weighted `v_0 + v_1` on the probe positions, for a randomly
initialised network and for a trained one:

| network | mean | mean abs | std |
|---|---:|---:|---:|
| untrained | -0.052 | 0.052 | **0.018** |
| trained 30 min | +0.052 | 0.058 | **0.044** |

Training does not cause the violation and does not cure it — a random network is
off by the same amount, and it even flipped sign. That is what an unconstrained
quantity looks like: nothing forces two independent outputs to negate at
initialisation, and no term in the loss asks for it afterwards, so it drifts.

But the split matters more than the magnitude. A constant offset added to both
players changes nobody's best response and is harmless. The *state-dependent*
part does change play, because it inflates some positions relative to others.
That is the part training makes **2.4x worse**, 0.018 to 0.044, and the part the
projection removes.

The targets are also root values of solves whose leaves are this same network,
so the violation feeds forward into what the network is then trained on. Worth stating because the obvious candidate is not the culprit:
the 180-degree mirror augmentation was *on* for `runs/base4h`, and it cannot fix
this. Mirroring maps a position to a different position with the seats
exchanged, which asks the network for equivariance between two states. This is a
constraint between the two players' values at *one* state, and nothing has ever
asked for it.

**The fix, stated correctly.** An earlier draft of this note said to subtract
half the violation from both players' values, which glosses over the structure:
there is no per-config pair to symmetrize. Player 0's leaf values are indexed by
player 0's configs and player 1's by player 1's, and they are counterfactual —
each is already weighted by the *opponent's* reach into the leaf. The
antisymmetry constraint is not per config, it is one linear equation per leaf on
the belief-weighted aggregates:

    m_0 = E_{b_0}[v_0],  m_1 = E_{b_1}[v_1],  and m_0 + m_1 should be 0.

So the projection is: form `s = m_0 + m_1` per leaf, subtract `s/2` from every
config's value for both players. That shifts a leaf's value *level* while
leaving the network's discrimination between configs within a player untouched
— and the level is exactly what CFR compares across a subgame. The
antisymmetric subspace contains the truth, so the projection cannot increase
error, and it makes `(m_0 - m_1)/2` an average of two estimates of one quantity
rather than one estimate.

Not implemented. It lives in `Solver::readout`, which is called per player off a
shared PBS-head pass and runs every CFR iteration, so it needs both players'
aggregates available at once — a real change to the hot loop, wanting
`train/test_parity.py` and `tests/rebel_solver.rs` in front of it rather than a
quick patch.

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
