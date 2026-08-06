# ReBeL for War Chest

An implementation of ReBeL — *Combining Deep Reinforcement Learning and Search
for Imperfect-Information Games*, Brown, Bakhtin, Lerer & Gong (NeurIPS 2020),
[arXiv:2007.13544](https://arxiv.org/abs/2007.13544) — on the War Chest engine
in this repo. The reference implementation it follows is
[facebookresearch/rebel](https://github.com/facebookresearch/rebel), whose
`csrc/liars_dice` is cited throughout.

Everything except the gradient step runs in Rust: the game, the belief filter,
the CFR subgame solves and the network forward passes. PyTorch ships weights
down and pulls tensors back once per epoch.

## 1. What is private, and why the belief state stays small

ReBeL's central object is the **public belief state** (PBS): the public state
plus a distribution over each player's private information. That is only usable
if the private space can be enumerated.

In War Chest two things are hidden:

* the **hand** — which coins were drawn this round;
* the **identities of face-down discards** — the coin spent on Pass, Claim
  Initiative, or a Recruit payment is never revealed.

Everything else is open: the board, supplies, recruits (revealed when taken),
face-up discards, eliminated coins, initiative state, and every *count*
(hand size, bag size, face-down count). So a player's private state is exactly
the pair `(hand, facedown)` — a `Config` — and the bag is derived:

```
bag = reserve - hand - facedown        reserve = bag + hand + facedown  (public)
```

The reserve is public because every action either moves a coin *within* it (a
face-down play) or out of it in full view (deploy, bolster, any maneuver, the
recruited coin). This is the load-bearing invariant: it is what lets one public
tree carry every config, and it is checked directly by
`features_do_not_leak_private_information`.

The draft fixes 4 unit types plus the Royal Coin, so at most `NSLOT = 5` coin
types per player; a hand holds at most 3. Measured over 41k positions of random
play, the reachable config set has **median 8, mean 34, p99 385** members. CFR
enumerates information states exactly — no particle approximation.

Beliefs are per player and independent (separate bags, no shared hidden
resource, unlike poker's shared deck), so a PBS factorises as
`(public state, belief_0, belief_1)` — the same shape as Liar's Dice in the
reference repo.

### Partially private actions

Pass, Claim Initiative and a Recruit payment announce the *event* but not the
coin, so several private actions collapse onto one public child of the tree. The
belief update is
`β'(c') ∝ Σ_{c, a : obs(a) = o, c →ᵃ c'} β(c) · π(a | c)`
— `obs_key` in `rebel.rs` defines the observation, `obs_child` in `search.rs`
carries the many-to-one map, and `play_game` does the sum. A Recruit reveals
which unit was taken and hides which coin paid for it.

### Chance

Round-start draws are the only chance nodes. Their outcome is private, so they
do not branch the public tree at all: the PBS transition is a deterministic
convolution of the belief with each config's own draw distribution
(`belief_after_draw`). When a bag empties, the whole discard pile is shuffled
in, which *erases* the face-down component — handled as a deterministic remap.
Bag emptiness is public, so every config reshuffles at the same moment.

The Warrior Priest (and V2) are **excluded from the draft pool**. Their
attribute triggers a *private* mid-round draw, which would add "which coin must
I now play" to the private state; the paper's own advice for such a case is to
clamp or exclude, and excluding keeps the config space exactly
`(hand, facedown)`.

## 2. Finite horizon and its payoff

ReBeL's theory needs a finite game. A game is capped at `MAX_MAIN_PLAYS = 256`
coin plays, and `plies_remaining` is part of the public state and a network
input — without it, PBS values near the cap are non-Markov and the value network
gets contradictory targets.

A flat zero at the horizon is a trap: under early, near-random play almost no
game ends by placing all six markers, so every target is zero and `V ≡ 0` is a
self-consistent fixed point with no gradient toward winning. The horizon is
instead scored on the win condition itself:

```
utility = ±1                                       (six markers placed)
utility = CAP_MARKER_VALUE · Δmarkers, = 0.15·Δ    (horizon reached)
```

Zero-sum, strictly inside ±1, dense from the first game, and it induces a
curriculum (take locations → deny locations → race to six). This is a change to
the terminal payoff of the game being solved, not a patch on the algorithm. It
distorts the equilibrium slightly, so the coefficient should be annealed toward
zero once horizon games become rare — the per-epoch `cap` column reports that
fraction.

## 3. Search: depth-limited CFR over PBSs

`engine/src/search.rs`. A node is a leaf when it is terminal, when it is a
chance node (clamping there keeps the subgame chance-free, the reference's own
v1 simplification), or at the depth limit. Leaf values come from the value
network.

Conventions follow `csrc/liars_dice` closely:

| | |
|---|---|
| solver | alternating-traverser CFR; the regret rule is a setting (below) |
| leaf value | `v_net(PBS)[hand(c)] × (opponent's unnormalised reach)` — counterfactual |
| network query | public features + both **normalised** reach vectors |
| initial strategy | uniform over legal actions, strategy sums seeded reach-weighted |
| value target | running mean of per-config root values over iterations |
| acting / belief propagation | the **current** regret-matching iterate, not the average |
| trajectory sampling | stop at a uniformly random iterate `t ~ U{0..T}`, act, then finish the solve before reading the target |
| exploration | `random_action_prob` for a uniformly sampled player, redrawn each decision |

Three things differ from poker:

1. **Action sets depend on the config** — you can only play a coin you hold.
2. **An action moves the information state**: hand minus that coin, and for a
   face-down play the coin lands in the face-down discard. `trans` carries the
   `(config, action) → config'` map, precomputed per node.
3. **Actions are partially private** (above).

### Which CFR

`Cfr { alpha, beta, gamma, predict }` in `search.rs`. Every variant worth
comparing is one formula with four numbers, which is why there is a parameter
struct here rather than five implementations. Discounted CFR (Brown & Sandholm
2019) multiplies accumulated *positive* regrets by `t^alpha / (t^alpha + 1)` and
negative ones by `t^beta / (t^beta + 1)` each iteration, and contributions to
the average strategy by `(t / (t + 1))^gamma`; `predict` is Predictive CFR+'s
optimism, which regret-matches on `R + predict * r` — the regret just observed
standing in for the one about to be seen.

| name | alpha | beta | gamma | predict | |
|---|---|---|---|---|---|
| `linear` | 1 | 1 | 1 | 0 | the reference implementation's, and the default |
| `plus` | inf | -inf | 2 | 0 | CFR+ (Tammelin 2014) |
| `dcfr` | 1.5 | 0 | 2 | 0 | DCFR — **what TurboReBeL itself runs** |
| `pcfr` | inf | -inf | 2 | 1 | PCFR+ (Farina, Kroer & Sandholm 2021) |
| `sapcfr` | inf | -inf | 2 | 1/3 | SAPCFR+ (Meng et al., AAAI 2026) |

`beta = -inf` zeroes negative accumulated regret, which is regret matching+;
`alpha = inf` leaves positive regret undiscounted. We shipped TurboReBeL's data
generation on top of linear CFR, so the solver is currently off-paper; §7 has
what that costs.

Regret matching floors the strategy at `1e-6` rather than at zero in every
variant. That is load-bearing, not cosmetic: `carried_beliefs` hands the
self-play walk one belief per iterate and the walk asserts each has the same
support as the live one, so a hard zero would drop configs and fail the assert.

**No handcrafted heuristic is injected into the search.** CFR starts uniform;
there is no
prior-biased regret initialisation from a handcrafted evaluation and no
heuristic action pruning, both of
which would bias the equilibrium by an amount nobody can measure. The greedy
knowledge enters through the value network, which is what CFR actually consumes.

## 4. The value network

The value of a leaf is a counterfactual value **per information state**, so it
is indexed by the same object beliefs, reaches and regrets are indexed by: the
config. `v̂(PBS, c) -> scalar`.

```text
  config tower:  z(c) = relu(phi(c) Wc + bc)                 [dg]
                 g(c) = z(c) Wg + bg                         [rank + 1]

  PBS tower:     hpub = relu(LN(x_pub W0 + b0)) W1
                 e_p  = sum_c beta_p(c) z(c)                 [dg] per player
                 h    = relu(LN(hpub + [e_0; e_1] Wb + b1))  [hidden]
                 u    = h Wu + bu                            [rank]

  value:         v(c) = <u, g(c)[..rank]> + g(c)[rank]
```

`phi(c)` is one config's exact counts: hand, face-down, and the derived bag,
plus the seat it belongs to. Sixteen numbers, nothing bucketed or averaged.
`config_features_separate_every_config` pins that two distinct private states
never share a feature vector, which is what makes `v̂` a function *of the
config* rather than of an equivalence class of them.

This is the reference implementation's shape, generalised. `csrc/liars_dice`
ends in `hidden -> num_hands`, which is exactly `<h, W2[:, c]> + b2[c]` — an
embedding *table* over private states, and `num_hands` also indexes its beliefs,
its regrets and its strategies. War Chest's private states do not fit in a table
(hand x face-down runs to ~145k and varies by draft), so the table becomes an
embedding *network* `g`. The belief is the same substitution on the input side:
instead of a fixed-length vector of per-private-state probabilities, it is the
belief-weighted sum of the same config embeddings.

**What is cached.** Inside a solve the same leaf is queried once per iteration
and only the beliefs move, so three things survive the whole solve: the public
tower `hpub` (the widest matmul in the network), and both `z(c)` and `g(c)` for
every config in the tree — a config's features do not depend on the iteration.
The same config recurs at hundreds of leaves, so the config tower runs once per
*distinct* config per solve (`Solver::intern_config`); the trainer deduplicates
a batch the same way, and it is the difference between a 0.5-second and a
4-second training step. What remains per iteration is one `2·dg -> hidden`
matmul per leaf, one LayerNorm, and one `rank`-long dot product per config.

`rank` is the one width that has to be chosen rather than inherited. The
reference gets `rank = hidden` for free because its readout is a lookup; here
every config costs a dot product of that length, per leaf, per iteration. A
config is sixteen numbers, so 64 is not a binding constraint on what the value
can depend on, and it is 6x less per-config work than the hidden width.

### What this replaced, and what it cost

The previous build keyed values by **hand** alone — 56 keys per player, with the
face-down composition entering as a belief-averaged marginal — and
`Data::push_value` projected the solver's per-config root values onto that basis
before training on them. §4 of this document used to call that "a coarseness
choice in `v̂`, not a soundness bug". That was wrong, and the error is worth
being precise about.

Configs sharing a hand have identical legal action sets, so if they also share
every leaf value they get identical regrets and therefore an identical strategy.
The coarse value function did not merely misvalue those situations: it made the
agent's strategy **measurable with respect to the hand**, so the agent could not
act on coins it had buried itself. A restricted strategy space is a different
game, and the fixed point being learned was that game's equilibrium.

Measured before the change (`examples/cfgvalue.rs`, 120 positions, uniform
beliefs over consistent configs, T=16):

| | greedy play | random play |
|---|---|---|
| configs per position / distinct hands | 3.5 / 2.5 | 137.3 / 18.1 |
| belief mass in hands holding >1 config | 9.5% | 60.8% |
| within-hand RMS bag deviation, per slot | 0.108 coins | 0.467 coins |
| same-hand config pairs getting identical play | 86% | 92% |

So the information was real — up to 1.34 coins of bag difference inside one hand
— and ~90% of the time the agent could not use it. The value error the
projection deleted was 0.0014 to 0.020 depending on depth, against a held-out
network error of ~0.09; small, and *growing with depth*, because the collapse
was only visible where a round-start draw fell inside the horizon. That is a
lower bound on itself and not the reason to fix it. The reason is that it was a
different game.

### Features

`rebel.rs::write_public_features` is public by construction: per-hex
occupancy/height/marker, per-player reserve/face-up/supply/eliminated counts,
slot identities, marker and count scalars, `plies_remaining`. Bag, hand and
face-down appear there *only* through their public sum (the reserve) and their
public sizes. `features_do_not_leak_private_information` checks it directly, and
`a_solve_reads_only_the_beliefs` checks the same property one level up — the
network is now asked about specific configs, so it matters that each query uses
only the *traverser's own* config, and that solving the same public position in
two different worlds gives bit-identical values and strategies.

Loss is a belief-weighted Huber over every config in the support: a config the
belief gives 1% to is worth 1% of the gradient, which is what makes the loss
match the distribution CFR actually queries. Targets are clipped to ±1, which is
where the true value function lives.

### The remaining approximation

The belief reaches the network as `sum_c beta(c) z(c)`, a fixed-width sum of
learned config embeddings. Unlike the value's dependence on a single config,
this one **cannot** be made exact at fixed width: a belief is a distribution
over a config space of ~145k, and the reference only escapes this because its
private space is small enough to feed verbatim. Two beliefs whose embeddings
coincide are indistinguishable to the network.

This does not change the game being solved — CFR carries exact beliefs
internally, and only the network's view of them is compressed — so it is
ordinary function-approximation error that shrinks with `dg` and with data. It
is still worth knowing the size of; see `TODO.md`.

## 5. Training

```bash
python train/train.py --minutes 30 --out runs/mine
```

Phase 1 — **greedy warm start** (`--warm-frac`). Both players are a stochastic
one-ply greedy bot on a public-information evaluation. Value targets blend that
evaluation, squashed into (-1, 1), with the realised game outcome
(`--eval-mix`): the outcome is unbiased but very noisy — most states genuinely
cannot predict the winner — while the handcrafted eval is biased but dense and
low-variance, which is what makes one-ply differences legible to CFR from the
first game.

ReBeL never plays a policy directly, so the value network is the natural
injection point for a starting behaviour; a heuristic *policy* prior would
additionally need the warm-start machinery of the paper's Appendix J just to
survive early CFR iterations. The network at the end of this phase is saved as
snapshot 0, labelled `init`: the zero point the Elo curve is read against. Its
bias washes out — every subsequent target comes from real solves and real
outcomes.

Phase 2 — **ReBeL**. Self-play with a CFR solve at every decision.

Training runs on the rulebook's recommended **starter matchup** by default
(Swordsman/Pikeman/Crossbowman/Light Cavalry vs Archer/Cavalry/Lancer/Scout);
`--random-draft` switches to randomised drafts, a training-distribution
extension with unit-set indicators already in the encoding.

Every `--snapshot-every` minutes the network is written to disk and training
continues. Nothing is compared, promoted or selected while the run is going.

Evaluation is one round robin at the end, `train/ladder.py`: every snapshot
against every other snapshot, against Greedy and against Random, scored with
Bradley-Terry into an Elo per player, with Random pinned at 0. What the run
reports is therefore a *curve* — strength against minutes trained — rather than
a number whose provenance depends on which checkpoint a mid-run match happened
to like.

This replaced a champion gate (AlphaGo Zero's rule: promote the live network
when it beats the reigning champion over 300 paired games). The gate was wrong
here in three ways. It spent training time on games — minutes per gate, on a
machine where minutes are the whole budget. Its standard error is ±0.029 at 300
games and ±0.046 at 120, which is the same size as or larger than the
improvement between two snapshots twenty minutes apart, so the promotions it
made were substantially draws of noise. And it answered a question nobody had:
"which single checkpoint should we ship" matters much less than "is this
training run making the agent stronger, and how fast", which a ratchet built out
of noisy pairwise tests cannot show at all. The ladder measures every snapshot
against every other one, at the end, where games are no longer competing with
training for the same eight cores.

Ladder matches are paired — the same draft and the same random stream for both
seatings — and use a full solve and the CFR average strategy.

### Measured result (30 minutes, 8-core M1, depth 2, T=64)

`runs/elo01`, 60 paired games per pairing, on the real game (horizon payoff 0),
Random pinned at 0:

| snapshot | trained | Elo |
|---|---|---|
| init (end of warm start) | 5 min | 356 ± 29 |
| s1 | 11 min | 748 ± 22 |
| s2 | 17 min | 842 ± 22 |
| s3 | 23 min | 852 ± 22 |
| final | 30 min | 852 ± 22 |
| Greedy | — | 174 ± 32 |
| Random | — | 0 ± 38 |

**The agent gains 392 Elo in the first six minutes of self-play, 94 in the next
six, and nothing measurable in the thirteen after that.** The plateau is the
result; the champion gate this replaced could not have shown it, because a
ratchet reports promotions rather than a shape.

Two checks on the ladder itself. Two snapshots taken 46 seconds apart rate three
points apart, inside the ±22 they are measured to, which is the noise floor
behaving as it should — nothing in the fit knows they are nearly the same
network. And the gap from `init` to `final`, 496 Elo, is the same quantity the
old `final_vs_init` score reported: `runs/cfgvalue01`'s 0.940 is 478 Elo. The
scale is new; the claim is not.

Earlier results, for continuity: three runs of the *hand-keyed* build scored
0.99-1.00 against Greedy and 0.925-0.960 against their own initial checkpoint,
and `runs/cfgvalue01` (10 minutes, T=16, the first config-keyed run) scored 0.961
and 0.940. None of these can be compared to the table above except through the
`init`-to-`final` gap: budget, iteration count and the replay sampler all changed
together.

Throughput on 8 cores in the ReBeL phase: **2.2 games/s** at T=64 with a full CFR
solve at every decision, against ~3.8 at T=16 — a factor of 1.6 for 4x the
iterations, because self-play stops at a uniformly random iterate and so averages
half the limit. The hand-keyed build managed ~14 games/s at T=8; per-config
values cost about 18% of generation rate (`docs/PERF.md`).

### The Monte-Carlo anchor, and why it is off

ReBeL's value target is purely bootstrapped. With depth-1 subgames that is
TD(0), and over a 250-ply game with a 150k replay buffer an early build
reliably found a self-consistent but *wrong* value function: in a first
30-minute run the training loss fell to 0.004 while the agent's score against
Greedy collapsed to 0.007. `--mc-mix` blends the realised return into the target
(TD(λ); MuZero's n-step bootstrap), and at 0.3 that collapse disappeared.

**That finding does not survive the current generation loop, and the default is
now 0.** Re-measured at matched wall-clock on the post-`PERF.md` code, mc-mix
0.3 is clearly *worse*: 0.837 against the initial checkpoint versus 0.937 for
pure bootstrap, 0.933 versus 0.978 against Greedy, and a training loss floor six
times higher (0.043 versus 0.007) because a realised return is a high-variance
label the network cannot fit. The original collapse was measured when a
10-minute budget bought 7 ReBeL epochs; it now buys over a hundred, and the
bootstrap has enough iterations to stay anchored on its own.

The mechanism that made the anchor necessary has not been proven gone, only
outrun — a run long enough to drift could still need it, so `--mc-mix` stays.

### What the value network is actually short of

Held-out error sat at RMS ~0.092 against a target spread of 0.39 and would not
move. Because a target is a deterministic function of the network's own input —
the CFR root value of the subgame at `(state, ctx, beliefs)`, which is exactly
what a row encodes — a dumped replay buffer is a noise-free
supervised dataset, and the question can be settled offline in minutes instead
of by training runs whose headline score wanders by ±0.05 (`train/offline.py`).

Three explanations were tested and two died:

* **Capacity.** No. Five architectures spanning 2.6x in trunk cost — the MLP of
  the day, a 512-wide MLP, and hex convolutions at 2 and 3 layers, 16 and 32
  channels — all landed within 4% of each other. (Measured on the hand-keyed
  network; the numbers in this section are all from before §4's rewrite and have
  not been re-measured against the config-keyed one.)
* **A bug in the targets.** No. `train/diagnose.py` finds rows whose inputs are
  byte-identical and compares their targets: rows recorded close together agree
  *exactly*. The encoding-to-target map is clean.
* **Target drift.** Real but small. The same duplicate analysis shows
  disagreement rising monotonically with how far apart two rows were recorded
  (0.0000 under 1k rows apart, 0.0234 beyond 50k), for 0.0145 RMS overall —
  2.5% of the variance of a 0.092 floor.

The answer was **data**. Training error sat at 2.7x below held-out error and
kept falling while held-out error stayed flat: the network was memorising. The
data-scaling curve confirms it and does not saturate — 40k rows give 0.0122,
80k 0.0103, 160k 0.0086, 284k 0.0082 — and at full data with augmentation the
train/test gap closes to zero.

Those two findings pull against each other, and the sampler is where they are
traded off. Data says hold as many distinct positions as memory allows; drift
says an old row's target was written by a network that has since moved, and is
wrong by up to 0.023 — a quarter of the held-out error — by the time the buffer
turns over. So a batch is a mixture: `--recent-mix` of it is drawn from the
newest `--recent-frac` of the buffer and the rest uniformly from all of it, which
at the defaults (0.5, 0.2) draws a row from the fresh slice six times as often as
an old one — three times the average rate — while leaving every row reachable. Pure recency was not tried and should not
be: it would discard exactly the distinct positions the scaling curve says are
the binding constraint.

Two more consequences. Replay capacity is an algorithmic knob, not a memory
setting.
And the 180-degree board symmetry is worth exploiting: rotating the board maps
white's starting locations exactly onto black's, so every position can be
presented a second way with the seats swapped, for free (`train/mirror.py`,
applied per batch so the buffer does not double). Measured: held-out loss
0.008446 → 0.008161 with the overfitting gap down 38%.

A note on the convolution, since it was the obvious thing to reach for. It does
match the MLP once augmentation removes the overfitting that was penalising its
extra parameters — but it is then *equalled by widening the MLP at 65% of the
cost*, 2.25x trunk compute for 1.5% of loss. On a spatial game with adjacency
rules that is a genuinely surprising result, and it is why the Rust kernel was
never written.

## 6. Tests

* `tests/rebel_pbs.rs::features_do_not_leak_private_information` — swapping a
  player's true config for any other config consistent with the same public
  counts must not move a single feature of the public encoding.
* `tests/rebel_pbs.rs::a_solve_reads_only_the_beliefs` — the same property one
  level up, and the one that matters now that the network is asked about
  specific configs: solve the same public position in two different worlds and
  the root values and the root strategy must agree bit for bit. Feeding both
  players' configs into a query would produce a perfect-information bot that
  looked spectacular; this is what would catch it.
* `tests/rebel_pbs.rs::config_features_separate_every_config` — two distinct
  private states must never share a feature vector, which is what makes the
  value a function of the config rather than of an equivalence class.
* `tests/rebel_pbs.rs::the_value_function_separates_configs_sharing_a_hand` —
  the regression test for the architecture this replaced: configs with the same
  hand and different face-down piles must get different values and different
  play. It was zero by construction before.
* `tests/rebel_solver.rs::subgame_solver_matches_tabular_cfr_on_micro_endgames`
  — the subgame solver against an independent vanilla CFR over world states,
  run under **every** regret rule. The game value of a two-player zero-sum game
  is unique, so any rule that converges must land on the same number; that makes
  one oracle the correctness net for the whole family. It also checks NashConv
  is non-negative and falls with iterations, and that reading a solve mid-flight
  leaves it able to continue.
* `tests/rebel_pbs.rs::belief_tracker_matches_brute_force` — the incremental
  tracker versus an exhaustive enumeration of every world consistent with the
  observation sequence, weighted by exact draw probabilities and the announced
  policy, agreeing to 1e-5 over tens of thousands of worlds. The brute-force
  side goes through the engine only; it never touches `Belief`,
  `advance_config` or `belief_after_draw`.
* `tests/scenarios.rs` (36 cases) and `tests/invariants.rs` — the engine itself,
  including that the horizon payoff is zero-sum and strictly inside ±1.

## 7. Known gaps

* **T = 64 CFR iterations**, against the paper's 256/1024.

  **Two corrections to everything below, August 2026.**

  *The numbers in this section were measured through a bug.* `value_under`
  left the tree's reach probabilities propagated under the reference strategy,
  and `update_regrets` assumes they are consistent with the current iterate and
  does not recompute them. Self-play never noticed — it calls `value_under`
  once, after the solve — but `solvererr.rs` reads a solve at each rung and then
  keeps iterating, so every rung after the first resumed from the wrong reaches.
  Both fixed-policy passes now restore them. The tables below have not yet been
  re-measured.

  *The question in this section was the wrong one.* It grades a solve by the
  distance from its own converged answer, which cannot compare two different
  regret rules, because each is graded against itself. The metric is now
  `Solver::nash_conv` — what a best response to the solve's own average strategy
  would gain, summed over the players — which is absolute, is zero exactly when
  the strategy is a fixed point of the subgame operator, and is what the CFR
  literature reports. Preliminary, 10 depth-2 positions with ~160-config
  supports: **DCFR at T=64 reaches the NashConv linear CFR needs T=512 for**
  (0.00035 against 0.00036). All four modern rules beat linear at every T and
  land within ~20% of each other. A full run is pending.

  The earlier default
  of 8 rested on micro-endgames solved against exact values (mean |error|
  0.0035), which converge almost immediately and understate the error on the
  ~540-node subgames self-play actually solves. Measured on real mid-game
  positions against a converged T=512 reference (`examples/solvererr.rs`), the
  root-value error is 0.0098 at T=8, 0.0036 at T=16 and 0.0016 at T=32 —
  8%, 3% and 1.3% of the spread of the values themselves, and stable across
  belief supports from 3 to 136 configs.

  The default was 16 for a while, on the following argument. What decides it is
  the
  *signed* mean error, not the absolute one. Zero-mean error behaves like noise:
  the network averages it away over millions of rows and it adds in quadrature
  with the network's own 23%-of-spread error, where a further 3% → 1.3% is
  invisible. A signed error is a bias that compounds every time the operator is
  applied and displaces the fixed point by roughly bias/(1 − γ), with γ near 1
  on an undiscounted 256-ply horizon. Measured:

  | T | mean \|err\| | signed mean | positions erring one way |
  |---|---|---|---|
  | 8 | 0.00952 | **+0.00149** | 50% |
  | 16 | 0.00355 | −0.00038 | 46% |
  | 32 | 0.00161 | −0.00042 | 48% |
  | 64 | 0.00079 | −0.00024 | 53% |

  T=8 carries a real one-directional component. By T=16 it is a tenth of the
  absolute error, has changed sign, and then stops moving — T=32 and T=64 sit at
  the same value, so what they remove is noise. The share of positions whose
  configs err in a consistent direction is ~50% at every T, exactly chance,
  which is what a noise term looks like. T=16 costs 36% of the target rate;
  T=32 costs 63%.

  **The default is now 64, and the argument above is the reason to distrust the
  argument, not the setting.** It reasons entirely about how a *fitted* network
  averages target error, and concludes that under-solving is free as long as its
  error is zero-mean. But the target is not data the network merely fits: it is
  the value of the subgame CFR was asked to solve, and at T=16 the subgame is
  one we stopped solving a fifth of the way in. The whole claim ReBeL makes is
  that a depth-limited *solved* subgame yields values consistent with the game's
  equilibrium. A cheap approximation to that solve gives up the property the
  method is built on in exchange for throughput, and throughput was never the
  scarce thing the agent's strength was bounded by — data was, and data is
  bounded by capacity as much as by rate. T=512 (the paper's regime) is worth a
  run of its own; 64 is the step taken first.

  Note that a training loss curve cannot be used to choose T: changing T changes
  the target function, so a lower loss at higher T may only mean the targets
  became easier to fit. Comparing those curves across T would mislead. The Elo
  ladder can, because it scores the resulting agents against a fixed reference
  and against each other; that comparison has not been run yet.
* **The subgame is not quite zero-sum.** Its leaves are network values, and
  nothing makes the network's value for player 0 at a leaf the negative of its
  value for player 1 there. So the game CFR is handed is only as zero-sum as the
  value network is antisymmetric — and CFR's convergence guarantee, and ReBeL's
  argument on top of it, both assume zero-sum.

  `nash_conv` reports the residual `v_0 + v_1` at the root for free. Measured on
  `t64_turbo_s14`'s final network, depth 2, 10 positions: **0.097 against a
  value spread of 0.128**. That is about what independent per-player errors of
  ~0.07 would produce, so it is the network's known held-out error showing up in
  a second place rather than a new defect — and it is a useful thing to have,
  because unlike a held-out loss it needs no reference and can be read during a
  run. It goes to exactly zero when every leaf is terminal, which is what the
  micro-endgame test sees.

  What has *not* been established is whether the residual matters: it may be
  harmless noise CFR averages over, or a consistent tilt that displaces the
  fixed point. Mirror augmentation already pushes toward antisymmetry
  indirectly. Symmetrising the value head — predicting `v_0` and taking
  `v_1 = -v_0` — would enforce it by construction and is the obvious experiment.
* **No policy network.** The paper treats it as optional (value net alone
  converges); it would be worth adding for CFR warm starting and for fast play.
* **The subgame is chance-free except for round-start draws**, which are walked
  through rather than branched (the reference's own v1 simplification for other
  chance nodes still applies: they are clamped as leaves).

## 8. Layout

```
engine/src/rebel.rs     PBS: configs, beliefs, observations, features
engine/src/search.rs    depth-limited CFR subgame solver
engine/src/selfplay.rs  self-play loop, belief filter, data collection, greedy bot, eval
engine/src/net.rs       batched inference MLP (Accelerate BLAS)
engine/src/py.rs        pyo3: set_weights / gen_data / eval_match
train/value_net.py      the value network, shared by everything that loads one
train/train.py          PyTorch training loop, snapshots on a timer
train/ladder.py         round robin over a run's snapshots -> Elo
train/plot.py           the four panels a run is read from
```
