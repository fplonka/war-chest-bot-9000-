# War Chest ReBeL — state of things and what happens next

Goal: a correctly implemented ReBeL for War Chest, faithful both to the game's
exact rules and to the algorithm as described in `papers/ReBeL_2007.13544v2.pdf`
and the reference implementation in `/tmp/rebel-ref`. The concrete bar is a
30-minute training run whose checkpoint beats both the initial checkpoint and
the Greedy bot, with the improvement visible *over the course of the run*.

---

## 1. What is verified

The game layer and the search layer are both checked against independent
implementations, not just against themselves.

| test | file | what it rules out |
| --- | --- | --- |
| `features_do_not_leak_private_information` | `engine/tests/rebel_pbs.rs` | Substituting a player's hidden config for any other config consistent with public information moves **zero** input features. No private information reaches the network. |
| `belief_tracker_matches_brute_force` | `engine/tests/rebel_pbs.rs` | The belief filter agrees with exhaustive enumeration of every consistent world to 1e-5, over up to 33k worlds. The oracle side uses only the engine's own transition function. |
| `subgame_solver_matches_tabular_cfr_on_micro_endgames` | `engine/tests/rebel_solver.rs` | Solver values agree with a completely separate vanilla CFR (world states, explicit infoset keys) to 4 decimal places on 10 non-trivial endgames; zero-sum residual exactly 0. Catches errors in reach propagation, the `(config, action)` transition map, the grouping of private actions under one public observation, and the counterfactual-value convention. |
| `cfr_iteration_count_bias` | `engine/tests/rebel_solver.rs` | Measures solve error against exact values as a function of iteration count. |

Plus 36 engine scenario tests and the playout invariants.

Two conclusions, with an important correction:

- **The subgame solver is exact.** It is not a source of training error.
- **16 CFR iterations is enough _at a given depth_.** Mean absolute value error
  is 0.0018 at T=16 versus 0.0002 at T=256 (and 0.017 at T=4).

  An earlier version of this document drew the conclusion "spending more compute
  on search will not help." That was wrong, and it closed off the branch that
  turned out to matter. The measurement was taken at `depth: 8` on micro
  endgames; it says nothing about depth. See §2b.

Private state is genuinely `(hand multiset, face-down discard multiset)`, with
`bag = reserve − hand − facedown` and `reserve` public and config-invariant.
Face-down discards are no longer treated as public — that was the original
correctness defect and it is gone.

With the fixed starter draft exactly 8 unit types plus the Royal Coins are in
play, so the two units excluded from the random-draft pool cost nothing in the
default configuration.

---

## 2. What went wrong, and why it was not what I first thought

The first plain-ReBeL run collapsed: training loss fell to 0.004 while the final
checkpoint scored **0.007 against Greedy** (lost 595 of 600 games).

I initially read this as TD(0) bootstrap collapse. It is not. The log shows a
sharp phase transition:

```
t=1341s  cap_frac 0.167   loss 0.00403   configs  8.1   decisions  6688
t=1423s  cap_frac 0.979   loss 0.00207   configs  4.1   decisions 12165
t=1798s  cap_frac 1.000   loss 0.00105   configs  4.7   decisions 12298
```

Every game pinned at the 256-ply horizon, decisions per epoch saturated at
`256 × games`, belief entropy collapsed.

**The agent learned to stall.** Each player has 6 markers and wins by placing
all 6, so the marker differential at the horizon runs up to ±5. At the
then-current `cap_marker_value = 0.15` that paid up to **±0.75 against a real
win of ±1.0**. Running out the clock with a marker lead was a risk-free way to
bank most of a win. ReBeL solved the game it was given, correctly. The game was
misspecified.

The loss fell to 0.004 not because the value function got good but because once
every game ends at the horizon, the value is a trivial public function of marker
counts.

This also explains why the earlier 30% Monte-Carlo blend appeared to work: it
anchored targets to realised outcomes, which suppressed the stalling attractor.
It was masking a reward bug, not fixing a learning bug. It has been removed.

---

## 2b. The bigger problem: `depth: 1` was not running ReBeL

Found by external review, confirmed by inspection at `search.rs:283`:

```rust
let leaf = s.is_terminal() || s.is_chance() || depth == 0;
```

At `depth: 1` the root's children are all built at depth 0, so every one is a
leaf. The root is the acting player's decision node. **The tree therefore
contains zero opponent decision nodes.** Measured over 1439 real MainPlay
positions with real filtered beliefs:

| depth | tree nodes | leaves | opponent decision nodes | ms/solve |
| --- | --- | --- | --- | --- |
| 1 | 21.9 | 20.9 | **0.00** | 0.2 |
| 2 | 385.2 | 366.9 | 17.28 | 2.2 |
| 3 | 6975.8 | 6661.7 | 19.88 | 29.1 |

Consequences: `update_regrets` only ever takes the `me == traverser` branch, so
the opponent's strategy never enters. `root_mean[acting]` becomes a linearly
weighted running mean of increasingly greedy 1-ply lookaheads over the network —
approximate value iteration with a `max`, carrying the usual overestimation bias
straight into a bootstrapped target. `root_mean[non-acting]` is a plain 1-ply
expectation of the network's own outputs, a pure TD(0) self-bootstrap. The
belief machinery only shapes leaf *inputs*; it does no game-theoretic work.

So the diagnostic runs in §5.1, which scored 0.81 against Greedy, were produced
by 1-ply lookahead value iteration over a belief-encoded state. A working agent,
but not ReBeL. The reference uses `max_depth: 2` for liar's dice.

Measured cost of the fix after removing duplicate actions (§4b): **18.7×**
(33.1 → 1.77 games/s generating; 8310 → 445 decisions/s). Evaluation is far
cheaper than generation at 8.2 games/s, so gating stays affordable.

## 3. Deviations from textbook ReBeL

The core loop is standard: self-play; at each decision solve a depth-limited
subgame with CFR over public belief states; a value network at the leaves;
train the network on the root values from the solve; act on a uniformly random
CFR iterate so the targets are unbiased (Theorem 3).

Four deviations, of which two are heuristics:

1. **Greedy warm start** *(heuristic, staying)*. Before ReBeL begins, the value
   network is trained on games played by a simple hand-written greedy bot. Not
   in the paper. Necessary because a uniformly random policy essentially never
   finishes a game, so there would be no signal to bootstrap from.

2. **256-ply cap with a marker-differential payoff at the horizon**
   *(heuristic, being removed)*. The cap itself is **required** — ReBeL needs
   finite games. The payoff at the cap is the training wheel, and it is the
   thing that broke run 1. See §4.

3. **Value network keyed by hand only.** The network predicts a value per *hand*
   (56 per player) rather than per full private config, with bag and face-down
   composition entering as marginal features. An approximation for tractability;
   CFR itself still uses exact configs.

4. **No policy network.** Full ReBeL also trains a policy net. Dropped — search
   alone selects moves. The paper treats the policy net primarily as an
   accelerator.

Nothing else is bolted on.

---

## 4. The horizon payoff: what changes and why

Three principles, applied.

**Magnitude must be strictly dominated by the real reward.** The best achievable
shaped outcome should sit far below the worst real win. Targeting a maximum
shaped payoff of about ±0.20 against a ±1.0 win gives **0.04 per marker**, down
from 0.15. (Default now `--cap-value 0.04`.)

**This shaping is not potential-based.** Ng, Harada and Russell (1999) show that
shaping of the form `γΦ(s′) − Φ(s)` provably preserves the optimal policy. A
bare terminal payoff at an artificial horizon does not — it defines a *different
game*. So the term should be treated as something to remove on a schedule, not
as a hyperparameter to tune to a good value. There is no good value.

**Anneal to zero early, linearly.** The payoff decays linearly to zero over the
first 40% of the ReBeL phase (`--anneal-frac 0.4`), leaving a long tail trained
purely on the real game, where the clock running out is a draw and only a real
win scores. Reaching zero only at the very end would mean shipping a network
fitted almost entirely to the wrong objective. Linear rather than exponential
because exponential lingers at small nonzero values, and small persistent bias
is exactly what bootstrapping amplifies.

**The annealer must not be reactive.** An earlier draft decayed the payoff when
horizon games became rare and *raised* it when they became common. That is
backwards: it would have reinforced the exact failure it was observing. The
schedule is now open-loop on wall-clock.

**Fixed (item 4 in the last exchange): the gate now always scores at
`cap_value = 0`.** Previously the periodic gate ran at whatever payoff was
currently live, so gate scores drifted with the anneal, were not comparable
across training, and biased checkpoint selection toward whichever weights best
exploited the shaped payoff. The generator's value is saved and restored around
the gate call.

---

## 4b. Rules and engine fixes from review

**Applied (all 43 tests pass, including both oracles):**

- **Cavalry tactic required no attack target** (`rules.rs`, `Tactic::Cavalry`).
  Every empty neighbour produced a `TacCavalryMove`; when no target followed,
  `advance` silently dropped the queued `CavalryAttack`, giving a successor
  state byte-identical to a plain `Move`. 5,690 duplicate pairs in 30,180
  decision nodes. Also a **rules error**: the FAQ (rules PDF p.17, LANCER)
  requires a legally attackable target to exist at the moment the tactic is
  chosen, and the Cavalry card has the same move-then-attack shape. Our Lancer
  already enforced this; the Cavalry did not. Now requires a target, checked on
  the pre-move board (exact, since the move only vacates our own hex and fills
  an empty one — neither can be an enemy target).

- **Light Cavalry two-step landing adjacent to the start** (`rules.rs`,
  `Tactic::LightCavalry`). Two axial steps with a 120° turn land at distance 1,
  where the plain `Move` is identical. Legal but redundant: 7,117 duplicate
  pairs, the single largest source. Now requires `dist[from][to] == 2`.

  Duplicate actions are not merely wasteful — they corrupt every
  uniform-over-actions distribution in the loop: CFR's initial strategy, the
  ε-exploration draw, `Agent::Uniform`, and the 5% floor in `greedy_policy`.
  They also make depth-2 subgames disproportionately expensive, since cost is
  quadratic in the branching factor.

- **Reachable non-terminal state with zero legal actions** (`state.rs`,
  `start_round_draws`). With every coin on the board, `begin_main_turn` sees
  both hands empty and calls `start_round_draws`, which queues no draws and sets
  `pending = MainPlay` — bypassing the `MAX_MAIN_PLAYS` guard, which only fires
  on *entry* to `begin_main_turn`. The result is a non-terminal state with no
  legal actions, which panics a rayon worker and kills the epoch. Random
  playouts never reach it; a long game with heavy bolstering does. Now sets
  `adjudicated_draw`.

**Queued, not yet applied:**

- **D5 — chance nodes are leaves the network is never trained on.** 8.8% of a
  MainPlay node's public children are `Draw` nodes, but `play_game` skips chance
  nodes without calling `push_value`, so the `pending_kind == Draw` one-hot
  column is never 1 in any training row and its weights never move from
  initialisation. Worse, at a Draw node both hands are empty, so the belief
  collapses to a point mass on `hand_index == 0` and the per-hand head
  degenerates to a single number — exactly at the round boundary, where
  face-down composition determines the next hand. The fix is better than adding
  chance-node targets: a draw's *public* projection is deterministic (hand +1,
  bag −1, no public branching), so walk straight through it with
  `belief_after_draw` and no tree branching. Removes the out-of-distribution
  leaves *and* lets depth-2 subgames span a round boundary.

- **S2 — LayerNorm.** The reference uses `use_layer_norm: true` with GELU
  (`conf/c02_selfplay/liars_sp.yaml`, `cfvpy/models.py`); ours is plain ReLU.
  Two feature normalisers also saturate: `bag_size` is divided by 12 but reaches
  18, the face-down count divided by 10 but reaches 14.

- **S4 — gating is under-powered.** At 120 games the standard error is ≈0.046,
  so a selected peak ~2σ above trend is likely noise; and reporting
  `final_vs_greedy` on the same quantity used for selection biases it upward.
  Raising `--gate-games`, and `final_vs_init` becomes the headline number.

**Deferred:** D6 (pending-continuation parameters are not encoded, so e.g.
`FootmanManeuver{hexes:[10]}` and `{hexes:[20]}` differ in 0 of 812 features
despite disjoint legal action sets — zero impact under the fixed starter draft,
small under `--random-draft`); S5 (ring buffer, only bites at `--cap 800000`);
S1/S6 (rare cross-version Footman tactic; a `Ctx::new` assert for the Warrior
Priest exclusion).

**Confirmed correct by review, worth recording:** the Theorem-3 sampling
procedure matches `recursive_solving.cc`; Linear CFR weights (regret ∝ t+1,
strategy ∝ t+2, `alpha = 2/(steps+2)`) match `subgame_solving.cc` exactly; the
counterfactual leaf convention `net[hand] × Σ opponent_reach` matches; and the
belief independence factorisation is *exactly* valid here, because bag size is
public, so reshuffle timing is public and identical across every config in the
support.

## 5. Immediate plan

### 5.1 Horizon payoff: settled, `--cap-value 0`

Two 12-minute runs (3 min greedy warm start, 9 min ReBeL, gating every 60s),
both at `depth: 1`:

| | A: no payoff | B: 0.04 annealed |
| --- | --- | --- |
| final vs Greedy | 0.807 | 0.810 |
| **final vs init** | **0.787** | 0.708 |
| final gate score | 0.863 | 0.846 |
| `cap_frac` | 0.02–0.17 | 0.00–0.19 |
| `probe_std` | flat ~0.40 | flat ~0.38 |
| `configs` | stable 18–20 | stable 18–21 |
| loss | 0.105 → 0.011 | 0.104 → 0.012 |

Same ceiling; A improves more over its own starting point. **The shaping is
redundant, not merely harmful-when-large** — the greedy warm start alone is
enough to make games finish, which was the shaping's only justification.

Going with `--cap-value 0`: one fewer heuristic, one fewer schedule, and it
sidesteps a problem the review raised about the anneal. At `--cap 800000` and
~7700 rows/epoch the buffer holds ~104 epochs ≈ 500 s of history, so targets
generated under the shaped payoff would outlive the schedule by ~40%, and during
the transition the buffer would hold a mixture of two different games with no
feature distinguishing them — worse than either endpoint.

Both runs are superseded by §2b: they measure 1-ply value iteration, not ReBeL.
They settle the payoff question, nothing more.

If games ever stop finishing, the fallback is a **shorter horizon** (150 plies),
not a payoff. A shorter cap changes the game but does not invent a second way to
*win*; 256 plies is ~42 rounds, well outside the observed distribution.

### 5.2 Game trace tooling (~50 lines, zero hot-loop cost)

Nothing currently exists for inspecting games; `Action` has no `Display`.

- `impl Display for Action` in `actions.rs` → `Deploy Swordsman a3`,
  `Attack b4 -> c4`, `Recruit (facedown) +Archer`.
- One pyo3 entry point `trace_game(seed, agents, ...) -> String` returning the
  move list with a periodic marker/bag summary.

Separate entry point, so no branch is added to the self-play hot loop and
throughput is unaffected. Worth also printing each side's belief entropy
alongside the moves — that is the quantity that collapsed in run 1, and seeing
it move is the cheapest early warning we have.

### 5.3 Order of work

1. ~~Cavalry / Light Cavalry duplicate actions, zero-legal-action crash~~ — done.
2. **Diagnostic C at `depth: 2`** — running. 12 min, `--cap-value 0`,
   `--steps 48` (holding the reference's ~4:1 train:generation ratio despite
   ~18× fewer rows per epoch), gating 200 games every 120 s. The question is
   whether real CFR at 1/18th the data beats 1-ply value iteration at full data
   within a fixed wall-clock budget. If it does not, the honest options are
   D5 (which makes depth-2 subgames span round boundaries and removes the
   out-of-distribution chance leaves) and a larger `--rebel-games` per epoch,
   not a retreat to depth 1.
3. D5, then S2 (LayerNorm), then S4 (gate sample size).
4. Trace tooling (§5.2).
5. The 30-minute run at `depth: 2`, with the gate curve as the evidence and
   `final_vs_init` as the headline.

---

## 6. Known-stale items

- `docs/REBEL.md` still documents `--mc-mix`, which no longer exists.
- `README.md` still carries the old "War Chest minus 2 of 19 units" framing,
  which is wrong for the fixed-draft default.

---

## 7. Measured throughput (8-core M1)

- ~900 games/s during the greedy warm start.
- ~11 games/s with a full CFR solve at every decision (~2000 solved
  decisions/s), ~25 configs per decision.
