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

**A side finding a hundred times larger than the one above.** The last table
`solvererr` prints is |v_0 + v_1|, how far the value network is from
antisymmetric. In a zero-sum game the two players' values must sum to zero. This
one sits at **0.0415 and does not move** across any iteration count or any rule.

That flatness is the diagnosis: a quantity CFR manufactured would change as CFR
converges. It does not, so the violation is in the network's leaf values and the
solve merely inherits it.

Measured directly as the network's own belief-weighted `v_0 + v_1` over 11,188
positions from 40 games, for a randomly initialised network and a trained one:

| network | mean | mean abs | std |
|---|---:|---:|---:|
| untrained | -0.055 | 0.057 | 0.033 |
| trained 30 min | **+0.025** | **0.032** | 0.032 |

Against a value spread of 0.416 that is about **8% of the signal**, and against
the network's own error of 0.099 on the same positions it is about **a third**.
Set beside the thing this sweep was actually about — the 0.00025 of target bias
from stopping CFR at T=64 — it is roughly 130 times larger.

**It comes from nowhere, which is the point.** A random network is off by the
same amount and with the opposite sign, so training neither creates the
violation nor removes it; it halves the mean and leaves the spread alone. That
is what an unconstrained quantity looks like. Nothing forces two independent
outputs to negate at initialisation, and no term in the loss asks for it after.

The obvious candidate is not the culprit: the 180-degree mirror augmentation was
*on* for `runs/base4h` and cannot fix this. Mirroring maps a position to a
*different* position with the seats exchanged, which asks for equivariance
between two states. Zero-sum is a constraint between the two players at *one*
state, and nothing has ever asked for it. The targets are also root values of
solves whose leaves are this same network, so the violation feeds forward into
what the network is next trained on.

**What was checked and cleared.** `State::utility` is exactly antisymmetric
(win +1 / loss -1 / horizon `cap * marker differential`). `eval_static` is
antisymmetric term by term and `eval_squashed` wraps it in an odd `tanh`.
`blend_outcome` flips sign for player 1 correctly, and `eval_mix` only applies
in the warm phase, so ReBeL targets are pure solve output. `mirror.self_check`
and `self_check_rows` pass on real data. The first player is randomised per
game and the draft is random, so the two seats are symmetric by construction —
and at 40 games they measure that way: configs 14.8/17.1, hands 1.92/1.96,
initiative 0.484/0.516, per-seat error 0.103/0.096 with biases +0.003/-0.003.

**A correction, and a lesson about sample size.** An earlier version of this
note reported the violation at +0.050 mean / 0.045 sd / 13% of signal, claimed
training made the state-dependent part 2.4x worse, and reported a 1.93:1 seat
imbalance in configs with player 1 fit measurably worse. All of that came from
`data/probe.npz`, which was built with `--games 2`. Its 1,334 positions are two
games' worth of highly correlated states, and every one of those numbers moved
or vanished at 40 games. The violation itself survived at roughly half the
size; the seat imbalance was noise and reversed sign. Build these sets with
tens of games, not two.

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

Implemented at the *target* level rather than the leaf level, as
`train.py::zero_sum` behind the `symmetrize` knob: the projection runs once when
a solve enters the replay buffer, where both players' belief-weighted values are
already to hand. Verified to take the violation to exactly zero, moving targets
by rms 0.035 against a spread of 0.321.

That is the cheap half. It cleans what the network is trained on, but the leaves
*inside* a solve are still the raw network, so the search still runs on a
slightly non-zero-sum game and the fix only reaches it through what the network
learns. The leaf-level version lives in `Solver::readout`, which is called per
player off a shared PBS-head pass every CFR iteration and would need both
players' aggregates at once — a hot-loop change wanting `train/test_parity.py`
and `tests/rebel_solver.rs` in front of it. Do that only if the cheap half pays.

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
