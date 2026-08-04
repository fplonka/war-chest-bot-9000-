# ReBeL for War Chest

An implementation of ReBeL — *Combining Deep Reinforcement Learning and Search
for Imperfect-Information Games*, Brown, Bakhtin, Lerer & Gong (NeurIPS 2020),
`papers/ReBeL_2007.13544v2.pdf` — on the War Chest engine in this repo.

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
| solver | alternating-traverser **linear CFR** |
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

**No heuristic is injected into the search.** CFR starts uniform; there is no
prior-biased regret initialisation and no heuristic action pruning, both of
which would bias the equilibrium by an amount nobody can measure. The greedy
knowledge enters through the value network, which is what CFR actually consumes.

## 4. The value network

A 2-hidden-layer MLP (`--hidden`, default 384), trained in PyTorch and evaluated
in Rust through Accelerate's `cblas_sgemm`, **split around the belief block**:

```
x_public (680) --W0--> h --LN,relu--> --W1--> \
                                               (+) --LN,relu--> --W2--> 2·NHAND
x_belief (132) ------------------------Wb---> /
```

Same parameter count and same depth as a plain `FEAT -> h -> h -> 2·NHAND`; the
only difference is *where* the belief block connects. It matters because of how
a subgame queries the network: the same leaf is evaluated once per CFR
iteration, and between iterations only its beliefs move — the public half of its
encoding is a property of the leaf's state and is fixed for the whole solve. So
the entire public tower (the widest matmul in the network, plus a full hidden
layer) is computed **once per leaf per solve**, and only the 132×h belief
projection and the output head run per iteration. That is 2.6x less
per-iteration matmul work, and it made the value function *better*, not worse
(§5).

`Mlp::trunk` computes the cached half; `Mlp::forward_split` the rest, emitting
only the one output head the current traversal reads.

**The deliberate information bottleneck.** In-subgame CFR uses exact
`(hand, facedown)` configs, but the network keys values by **hand** (56 keys per
player), with bag and face-down composition entering as marginal features.
Configs sharing a hand share a leaf value. This is a coarseness choice in `v̂`,
not a soundness bug — CFR-D stays well defined under a coarser value function —
and it is what keeps leaf queries affordable in the inner loop. Root training
targets are belief-weighted averages over configs per hand key
(`Data::push_value`). If value loss ever plateaus high, the escape hatch is to
widen the key to `(hand, bucketed bag composition)`.

Features (`rebel.rs::write_features`) are public by construction: per-hex
occupancy/height/marker, per-player reserve/face-up/supply/eliminated counts,
slot identities, marker and count scalars, `plies_remaining`, and the two belief
blocks (hand-key distribution + bag and face-down composition marginals). Bag,
hand and face-down appear *only* through their public sum and their public
sizes.

Loss is Huber over the supported hand keys; targets are clipped to ±1, which is
where the true value function lives.

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
`ckpt_init.pt`, the **initial checkpoint**. Its bias washes out: every
subsequent target comes from real solves and real outcomes.

Phase 2 — **ReBeL**. Self-play with a CFR solve at every decision.

Training runs on the rulebook's recommended **starter matchup** by default
(Swordsman/Pikeman/Crossbowman/Light Cavalry vs Archer/Cavalry/Lancer/Scout);
`--random-draft` switches to randomised drafts, a training-distribution
extension with unit-set indicators already in the encoding.

Every `--gate-every` seconds the live network plays a short match against the
fixed Greedy reference and the best-scoring weights are kept; that checkpoint is
what gets saved. Bootstrapped value learning is not monotone — the gate curve
below wanders by ±0.1 — so selecting on measured strength rather than on
"whatever was live when the clock ran out" is both standard practice and the
difference between a usable result and a coin flip.

Evaluation plays paired matches — same draft and the same random stream for both
seatings — using a full solve and the CFR average strategy.

### Measured result (10 minutes, 8-core M1, depth 2, `runs/perf04_final`)

```
final checkpoint  vs Greedy              score 0.99 - 1.00
final checkpoint  vs initial checkpoint  score 0.92 - 0.96
```

200 paired games per pairing, on the real game (horizon payoff annealed to 0).
Three runs of the same configuration span 0.99-1.00 and 0.925-0.960; the spread
is the run-to-run noise of a 10-minute budget, not a difference between builds.

Throughput on 8 cores in the ReBeL phase: **~11.5 games/s** with a full CFR
solve at every decision (~1400 solved decisions/s), ~24 configs per decision.
That is 9-10x what this took before `docs/PERF.md`'s work, which is what turns a
10-minute budget from 7 ReBeL epochs into ~120.

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
what `write_features` encodes — a dumped replay buffer is a noise-free
supervised dataset, and the question can be settled offline in minutes instead
of by training runs whose headline score wanders by ±0.05 (`train/offline.py`).

Three explanations were tested and two died:

* **Capacity.** No. Five architectures spanning 2.6x in trunk cost — the current
  MLP, a 512-wide MLP, and hex convolutions at 2 and 3 layers, 16 and 32
  channels — all landed within 4% of each other.
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

Two consequences. Replay capacity is an algorithmic knob, not a memory setting.
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
  counts must not move a single feature. This is the direct check that the bot
  cannot read its opponent's hidden coins.
* `tests/rebel_pbs.rs::belief_tracker_matches_brute_force` — the incremental
  tracker versus an exhaustive enumeration of every world consistent with the
  observation sequence, weighted by exact draw probabilities and the announced
  policy, agreeing to 1e-5 over tens of thousands of worlds. The brute-force
  side goes through the engine only; it never touches `Belief`,
  `advance_config` or `belief_after_draw`.
* `tests/scenarios.rs` (36 cases) and `tests/invariants.rs` — the engine itself,
  including that the horizon payoff is zero-sum and strictly inside ±1.

## 7. Known gaps

* **T = 16 CFR iterations**, against the paper's 256/1024. The earlier default
  of 8 rested on micro-endgames solved against exact values (mean |error|
  0.0035), which converge almost immediately and understate the error on the
  ~540-node subgames self-play actually solves. Measured on real mid-game
  positions against a converged T=512 reference (`examples/solvererr.rs`), the
  root-value error is 0.0098 at T=8, 0.0036 at T=16 and 0.0016 at T=32 —
  8%, 3% and 1.3% of the spread of the values themselves, and stable across
  belief supports from 3 to 136 configs.

  **16 is where the systematic component of that error disappears, which is why
  it is the setting and why 32 and 64 are not.** What decides this is the
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
  T=32 costs 63% to buy nothing that survives averaging.

  Note that a training loss curve cannot be used to choose T: changing T changes
  the target function, so a lower loss at higher T may only mean the targets
  became easier to fit. Comparing those curves across T would mislead.
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
train/train.py          PyTorch training loop
```
