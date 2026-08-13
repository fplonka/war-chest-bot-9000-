# ReBeL for War Chest

What the agent is, as it stands. Design history and superseded measurements are
in the git log, not here.

Papers: ReBeL (Brown, Bakhtin, Lerer & Gong, NeurIPS 2020,
[arXiv:2007.13544](https://arxiv.org/abs/2007.13544)); TurboReBeL (ICLR 2026
submission) for the data generation; Discounted CFR (Brown & Sandholm 2019) and
Predictive CFR+ (Farina, Kroer & Sandholm 2021) for the regret rules. The
reference implementation is `facebookresearch/rebel`, whose `csrc/liars_dice`
the conventions follow.

Everything except the gradient step runs in Rust: the game, the belief filter,
the CFR solves and the network forward passes. PyTorch ships weights down and
pulls tensors back once per epoch.

## 1. The public belief state

Two things are hidden in War Chest:

* the **hand** — which coins were drawn this round;
* the **identities of face-down discards** — the coin spent on Pass, Claim
  Initiative or a Recruit payment is never revealed.

Everything else is public, including every *count*. So a player's private state
is exactly the pair `(hand, facedown)` — a `Config` — and the bag is derived:

```
bag = reserve - hand - facedown        reserve = bag + hand + facedown  (public)
```

The reserve is public because every action either moves a coin *within* it (a
face-down play) or out of it in full view. That invariant is what lets one public
tree carry every config, and `features_do_not_leak_private_information` checks
it.

A draft fixes 4 unit types plus the Royal Coin, so at most `NSLOT = 5` coin types
per player; a hand holds at most 3, even across a Warrior Priest draw — the
trigger is always preceded, in the same play chain, by the coin play that
fired it. Over 120k positions of random play with the full draft pool the
reachable config set has median 22, mean 57, p99 567. CFR enumerates information
states exactly — no particle approximation.

Beliefs are per player and independent (separate bags, no shared hidden
resource), so a PBS factorises as `(public state, belief_0, belief_1)`.

**Partially private actions.** Pass, Claim Initiative and a Recruit payment
announce the event but not the coin, so several private actions collapse onto one
public child. The belief update is
`β'(c') ∝ Σ_{c, a : obs(a) = o, c →ᵃ c'} β(c) · π(a | c)` — `obs_key` in
`rebel.rs` defines the observation, `obs_child` in `search.rs` carries the
many-to-one map, `play_game` does the sum. A Recruit reveals which unit was taken
and hides which coin paid.

**Chance.** Round-start draws are the only chance nodes. Their outcome is
private, so they do not branch the public tree: the PBS transition is a
deterministic convolution of the belief with each config's draw distribution
(`belief_after_draw`). When a bag empties the discard pile is shuffled in, which
erases the face-down component; bag emptiness is public, so every config
reshuffles at the same moment.

**The Warrior Priest pair (units 18 and 54) is in the draft pool.** Its
attribute triggers a private mid-round draw, so the private state is the
triple `(hand, facedown, inflight)` — the drawn coin waiting to be played.
The in-flight coin is transient: it is set by the draw, cleared when the forced
play resolves, and is always absent at a network-query boundary, so it never
enters a replay row or the encoding.

## 2. Horizon

A game is capped at `MAX_MAIN_PLAYS = 256` coin plays, and `plies_remaining` is
part of the public state and a network input. The horizon is scored on the win
condition:

```
utility = ±1                       (six markers placed)
utility = CAP_MARKER_VALUE · Δmarkers    (horizon reached)
```

Zero-sum and strictly inside ±1. The coefficient is annealed to zero over a run;
the per-epoch `horizon` column reports the fraction of games reaching it.
Historical logs called this column `cap`; it never measured the search-tree
node cap.

## 3. Search

`engine/src/search.rs`. A depth-limited CFR solve over the public tree rooted at
a PBS. A node is a leaf when it is terminal or at the depth limit; depth counts
completed coin plays, so the micro-decisions inside one tactic ride free. A
round-start draw is walked through rather than branched. Leaf values come from
the value network.

| | |
|---|---|
| solver | alternating-traverser CFR; the regret rule is a setting (below) |
| leaf value | `v_net(PBS)[c] × (opponent's unnormalised reach)` — counterfactual |
| network query | public features + both **normalised** reach vectors |
| initial strategy | uniform; strategy sums seeded reach-weighted |
| acting / belief propagation | the **reference strategy** — the CFR average at the end of the solve |
| trajectory sampling | stop at a uniformly random iterate, act, then finish the solve before reading the target |
| exploration | `random_action_prob` for a uniformly sampled player, redrawn each decision |

Three things differ from poker: action sets depend on the config; an action moves
the information state (`trans` carries `(config, action) → config'`); actions are
partially private.

**No handcrafted heuristic enters the search.** CFR starts uniform. There is no
prior-biased regret initialisation from `eval_static` and no heuristic pruning.

### Which CFR

`Cfr { alpha, beta, gamma, predict }`. Discounted CFR's family covers every
variant worth comparing, so this is four numbers rather than five
implementations: accumulated *positive* regrets are multiplied by
`t^alpha / (t^alpha + 1)` each iteration, negative ones by
`t^beta / (t^beta + 1)`, and contributions to the average strategy by
`(t / (t + 1))^gamma`. `predict` is Predictive CFR+'s optimism: regret matching
runs on `R + predict · r`, the regret just observed standing in for the one about
to be seen.

| name | alpha | beta | gamma | predict |
|---|---|---|---|---|
| `linear` | 1 | 1 | 1 | 0 |
| `plus` (CFR+) | inf | -inf | 2 | 0 |
| `dcfr` | 1.5 | 0 | 2 | 0 |
| `pcfr` (PCFR+) | inf | -inf | 2 | 1 |
| `sapcfr` (SAPCFR+) | inf | -inf | 2 | 1/3 |

`beta = -inf` zeroes negative accumulated regret, which is regret matching+;
`alpha = inf` leaves positive regret undiscounted. The default is `linear`.

Regret matching floors the strategy at `1e-6` rather than at zero in every
variant. `carried_beliefs` hands the self-play walk one belief per iterate and
the walk asserts each has the same support as the live one, so a hard zero would
drop configs and fail that assert.

### Measuring a solve

`Solver::nash_conv` returns two numbers.

* `nash` — `Σ_p (BR_p − v_p)`, what a best response to the reference strategy
  would gain. Zero exactly when the strategy is an equilibrium of the subgame it
  induces. Absolute, so it compares regret rules against each other.
* `zero_sum` — `v_0 + v_1` at the root, which is **not** zero: the leaves are
  network values, and nothing makes the network's value for player 0 at a leaf
  the negative of its value for player 1. The subgame is only as zero-sum as the
  network is antisymmetric. It vanishes when every leaf is terminal.

Both freeze the leaf values at the ones the reference strategy induces, so this
is exploitability of the depth-limited game the reference defines, not of War
Chest. `examples/solvererr.rs` sweeps regret rule × iteration count.

## 4. Data generation

TurboReBeL's single-sample multi-iteration generation. One solve yields a
training row per kept iterate instead of one row, all valued under the same
reference strategy, so raising the iteration count stops costing data rate.

* **Phase 1** runs the full solve.
* **Phase 2** (`Solver::value_under`) computes the root value per config under
  the reference strategy, once per carried belief.
* `Solver::carried_beliefs` returns the belief at the walk's exit leaf under each
  kept iterate; those become the next solve's roots.

Snapshots are thinned to the log-spaced iterates (0, 1, 2, 4, 8, …) plus the
final one — the spread is in the early iterations — so a solve contributes ~9
rows, not T+1. Rows within a solve are therefore not near-duplicates and the
replay buffer samples rows uniformly; only the train:generation step count counts
solves.

## 5. The networks

`train/value_net.py` defines them; `engine/src/net.rs` runs them. `dims` is
`[pub, hidden, head, cfeat, dg, rank, afeat, de, dc, enc]` and is the single
source of truth for every buffer size — `head` is the width of the second
public matrix, the belief projection, the second LayerNorm and both readouts
(`head == hidden` is the pre-split network), and `enc` names the board
encoder (0 = flat; the hex-neighbour candidate is Python-only until a screen
picks it). `Mlp.flat()` produces the flat arrays both `set_weights` and
`export_weights.py` ship, so there is one place for Python and Rust to agree, and
`train/test_parity.py` checks it.

### The card describer

Nothing in the network names a *unit*. Each of the ten coin types in play — five
per player — is summarised by its 25 rulebook facts, and everything that refers
to a card refers to a coin-type index into that table:

```text
  e = relu(card facts Wd0 + bd0) Wd1 + bd1 + id_emb(unit)    [NTYPE, de]
```

Built once per solve: the draft is fixed for the game. The hex block, the pile
summary, the holding tower and the action tower all read a row of it, through a
one-hot over the ten coin types. That is what makes a draft the network has
never seen describable rather than an unknown identity code, and
`card_features_separate_every_draftable_unit` pins the precondition — if two
cards shared a fact vector the describer would merge them silently.

**A stored row holds one-hots, not embeddings.** `e` is learned, so a replay row
that contained it would carry whichever weights were live when it was written,
go stale as training moved them, and pass no gradient back to the describer. The
row keeps raw facts and the network does the lookup, which also keeps
`write_public_features` a pure function of the position.

The pile summary and the holding tower are **sums over coin types**. A sum has no
order, so nothing depends on which slot a card landed in and any draft fits.

### Value

The value of a leaf is a counterfactual value **per information state**, so it is
indexed by the config: `v̂(PBS, c) -> scalar`.

```text
  holding tower: z(c) = sum_k relu([counts_k, seat, e_k] Wc + bc)   [dg]
                 z(c) += relu(z Wh1 + bh1) Wh2 + bh2  (residual; Wh2 starts 0)
                 g(c) = z(c) Wg + bg                                [rank + 1]

  trunk input:   x    = [hex facts | e of each occupant | pile summary | loose]
  PBS tower:     hpub = relu(LN(x W0 + b0)) W1               [head]
                 b_p  = sum_c beta_p(c) z(c)                 [dg] per player
                 h    = relu(LN(hpub + [b_0; b_1] Wb + b1))  [head]
                 u    = h Wu + bu                            [rank]

  value:         v(c) = <u, g(c)[..rank]> + g(c)[rank]
```

The holding tower rectifies *before* the sum. A sum of raw linear maps is a
linear map of the sum, and the sum of the inputs has forgotten which count
belongs to which card — the one thing the tower exists to remember.

`config_features_separate_every_config` pins that two distinct private states
never share a feature vector.

This is the reference implementation's shape with its fixed-width private-state
dimensions replaced by learned functions, because War Chest's private states do
not fit in a table (hand × face-down runs to ~145k and varies by draft). The
belief is the same substitution on the input side.

The belief reaching the network as `sum_c beta(c) z(c)` is the one place a fixed
width is a real approximation: a belief is a distribution over a config space too
large to enumerate. It does not change the game being solved — CFR carries exact
beliefs internally — only the network's view of it.

### Policy

```text
  q(a)       = relu(psi(a) Wq + bq)                          [rank]
  logit(a,c) = <(h Wp + bp) + (z(c) Wk + bk), q(a)>
```

Three matrices, both towers shared with the value. `psi(a)` describes an action:
kind, the three squares it names, the coin slot that pays, the slot a Recruit
takes, and whether the coin goes face down — all one-hot with an explicit
"absent" slot. `action_features_separate_every_action` pins that two distinct
actions never share a description.

Labels are free: the solve's own average strategy at the root. The label attaches
to the solve's live-belief row — the other rows of a solve share one public state
and differ only in belief, so they would all carry the same strategy.

Checkpoints written before the policy head loaded with `strict=False` leave these
three matrices at their initialisation; `has_policy` records that, and anything
reading the policy asserts on it. Checkpoints from before the card describer do
not load at all — the trunk's input is a different width.

### Auxiliary heads

Training only, one matrix off the same hidden layer: each player's markers on the
board three rounds later, whether initiative changes hands next round, and the
result as three classes. Backfilled from a per-round timeline the game records as
it runs. They are dense — every row gets a different answer, unlike the single
value number — and they are never in `flat()`, so the Rust play path never sees
them and they cost nothing at inference. The trainer does not train them.

### Warm start

`Solver::warm_start` seeds a solve from the policy head instead of from a uniform
strategy: start CFR as though the policy had already been played for `--warm`
iterations. One traversal under the policy gives the instantaneous regret it
accrues, `r(a) = v(a) − Σ_a π(a) v(a)`; scaling that by the weight is the whole
of it, and seeding the average strategy the same way keeps the two consistent.

The baseline has to be the value of *playing the policy*. Using the
best-response value instead — `v(a) − max_a v(a)` — is non-positive everywhere
and zero at the best action, so regret matching clamps every action to the floor
and hands back a uniform strategy, destroying exactly what the seed exists to
inject.

`a_warm_start_does_not_move_the_fixed_point` pins the property that matters: the
subgame's value is unique, so a warm-started solve and a cold one must agree once
both converge. A seed that changed the answer would be biasing it rather than
accelerating it, and a strength gate could not tell those apart.

Default 0 — off. `examples/solvererr.rs` takes a warm-start weight as its sixth
argument and reports every regret rule cold and warm side by side; the decision
rule is whether warm at `T/2` beats cold at `T`.

Measured so far, on a head with 2.5 minutes of training: warm is worse than cold
at every rung, monotonically in the seed weight, and converges to the same place.
That is what a *correct* seed carrying a *weak* policy looks like — this head is
worth less than four CFR iterations, so injecting it as fifteen costs accuracy.
A4 is blocked on a policy head worth more than that.

### The pre-describer encoder

`engine/src/v1.rs` and `train/value_net_v1.py` are the encoding and network a
checkpoint from before the card describer was trained with, frozen and
eval-only. Keyed off `dims` (five entries against the current ten); a solve
takes its encoder from the net it was handed rather than from a constant.

They exist for one reason: the describer changed the public encoding's width, and
a gate that cannot play the new architecture against the pool cannot answer the
only question a gate is for. Nothing here is maintained or extended — delete both
when the pool has rotated past every checkpoint that needs them.

### What is cached

Inside a solve only the beliefs move, so three things survive it: the public
tower `hpub`, and `z(c)` and `g(c)` for every config in the tree. The config
tower runs once per *distinct* config per solve (`Solver::intern_config`); the
trainer deduplicates a batch the same way. What remains per iteration is one
`2·dg -> head` matmul per leaf, one LayerNorm, and one `rank`-long dot
product per config.

### Features

`rebel.rs::write_public_features` is public by construction: per-hex
occupancy/height/marker, per-player reserve/face-up/supply/eliminated counts,
slot identities, card properties, marker and count scalars, `plies_remaining`.
Bag, hand and face-down appear only through their public sum and their public
sizes.

Loss is a belief-weighted Huber over every config in the support, so a config the
belief gives 1% to is worth 1% of the gradient. Targets are clipped to ±1.

The board's 180-degree symmetry is used as augmentation: rotating the board maps
white's starting locations onto black's, so every position can be presented a
second way with the seats swapped (`train/mirror.py`, applied per batch).

## 6. Training

```bash
python train/train.py --minutes 30 --out runs/mine
```

**Phase 1 — greedy warm start** (`--warm-frac`). Both players are a stochastic
one-ply greedy bot on a public evaluation. Value targets blend that evaluation,
squashed into (-1, 1), with the realised outcome (`--eval-mix`). The network at
the end of this phase is snapshot 0, labelled `init`.

**Phase 2 — ReBeL.** Self-play with a CFR solve at every decision. Default
`--iters 64`, `--depth 2`.

**The policy head** is not trained. The weights exist so a later gate can turn
them on without a shape change. A value target is bootstrapped and gains from
being averaged over a long history; a strategy is not, and the epoch regenerates
every one of them. Its label attaches to the solve's **live-belief row** — the
rows of one solve share a public state and differ only in belief, while the
reference strategy is a single object, so labelling all of them would teach the
head that the belief does not matter.

Training runs on the rulebook's starter matchup by default;
`--random-draft` randomises the draft.

**The replay sampler.** `--recent-mix` of a batch is drawn from the newest
`--recent-frac` of the buffer and the rest uniformly from all of it, which at the
defaults (0.5, 0.2) draws a fresh row six times as often as an old one while
leaving every row reachable. Old rows carry targets written by a network that has
since moved; the per-epoch `old=`/`new=` columns report the gap.

`--mc-mix` blends the realised game outcome into the value target (TD(λ)).
Default 0.

**Evaluation** is one round robin at the end, `train/ladder.py`: every snapshot
against every other, against Greedy and against Random, fitted with
Bradley-Terry into an Elo each, Random pinned at 0. Matches are paired — the same
draft and random stream for both seatings — and use a full solve and the
reference strategy. `ladder.py` accepts snapshots from several run directories so
runs can be rated on one scale.

Nothing is compared, promoted or selected while a run is going.

## 7. Tests

* `rebel_pbs.rs::features_do_not_leak_private_information` — swapping a player's
  true config for any other consistent with the same public counts must not move
  a feature.
* `rebel_pbs.rs::a_solve_reads_only_the_beliefs` — solve the same public position
  in two different worlds; root values and root strategy must agree bit for bit.
* `rebel_pbs.rs::config_features_separate_every_config` and
  `action_features_separate_every_action` — distinct configs, and distinct
  actions, never share a feature vector.
* `rebel_pbs.rs::the_value_function_separates_configs_sharing_a_hand` — configs
  with the same hand and different face-down piles get different values and
  different play.
* `rebel_pbs.rs::belief_tracker_matches_brute_force` — the incremental tracker
  against an exhaustive enumeration of every world consistent with the
  observation sequence, to 1e-5 over tens of thousands of worlds. The brute-force
  side goes through the engine only.
* `rebel_solver.rs::subgame_solver_matches_tabular_cfr_on_micro_endgames` — the
  solver against an independent vanilla CFR over world states, under every regret
  rule. The game value is unique, so all must agree. Also checks NashConv is
  non-negative, falls with iterations, and that reading a solve mid-flight leaves
  it able to continue.
* `train/test_parity.py` — the Rust network against PyTorch on the same weights.
* `scenarios.rs` (36 cases) and `invariants.rs` — the engine itself.

## 8. Tools

* `train/offline.py` — fits candidate architectures to a frozen replay dump.
  Same data, same targets, so architectures compare exactly.
* `train/diagnose.py` — how learnable a dump's targets are, model-free.
* `examples/solvererr.rs` — regret rule × iteration count, by NashConv and by
  target error against a converged reference.
* `examples/featstats.rs` — the real range of every feature.
* `examples/cfgvalue.rs` — how far the value separates configs.
* `train/ladder.py` — Elo over snapshots plus Greedy and Random.
* `train/test_parity.py` — the Rust network against PyTorch, per seam.

## 9. Layout

```
engine/src/rebel.rs     PBS: configs, beliefs, observations, features
engine/src/search.rs    depth-limited CFR subgame solver
engine/src/selfplay.rs  self-play loop, belief filter, data collection, greedy bot, eval
engine/src/net.rs       batched inference (Accelerate BLAS)
engine/src/py.rs        pyo3: set_weights / gen_data / eval_match
train/value_net.py      the networks, shared by everything that loads one
train/train.py          PyTorch training loop, snapshots on a timer
train/ladder.py         round robin over snapshots -> Elo
train/report.py         the panels a run is read from
```
