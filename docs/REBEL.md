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
is the triple `(hand, facedown, inflight)` — a `Config` — and the bag is derived:

```
bag = reserve - hand - facedown - inflight
reserve = bag + hand + facedown + inflight  (public)
```

`inflight` is the Warrior Priest's drawn coin, waiting to be played. Its size is
public (the pending node says a forced play is owed); its identity is not. Empty
at every MainPlay, so it never enters a training row.

The reserve is public because every action either moves a coin *within* it (a
face-down play) or out of it in full view. That invariant is what lets one public
tree carry every config, and `features_do_not_leak_private_information` checks
it.

A draft fixes 4 unit types plus the Royal Coin, so at most `NSLOT = 5` coin types
per player; a hand holds at most 3. The Warrior Priest's draw does not join the
hand — it waits in flight — so it cannot push the cap. Over 120k positions of
random play with the full draft pool the
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
`alpha = inf` leaves positive regret undiscounted. The default is `dcfr`.

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
Chest.

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

## 5. The value network

`train/value_net.py` defines it; `engine/src/net.rs` runs it. The shape is
fixed: `dims` is `MODEL_TAG = [5]` and `NetLayout::new` refuses anything else, so
the job contract carries it as a version tag rather than as a set of widths,
and there is no checkpoint from another architecture to accommodate.
`Net.flat()` writes the flat arrays that `set_weights` and
`export_weights.py` ship, `NetLayout::new` reads them back, and
`train/test_parity.py` holds the two implementations to the same answers.

### The pieces, split by how often CFR runs them

CFR re-asks every leaf of the subgame on every iteration. At depth 2 and `T=64`
a solve has ~1,015 leaves and runs the join 78 times for each of the two seat
views, so ~1,015 board evaluations against ~158,000 belief-conditioned ones.
Capacity goes where it is amortised and the per-iteration path stays thin. The
trunk sees the physical board, which is the same for both seats; what makes a
view is the belief order and the seat scalar the join reads.

| piece | runs | produces |
|---|---|---|
| card describer, `Net::cards` | twice per solve | one token per coin type |
| trunk, `Net::board` | once per physical leaf per solve | the board vector `P` |
| config encoder, `Net::configs` | once per distinct config | `f(c)`, `g(c)` |
| join, `Net::join` | every CFR iteration | `h` |
| readout, `Net::values` | every CFR iteration | `v(c) = <f(c), h> + bias` |

DeepStack and ReBeL get this split for free: their tower runs once per public
leaf and emits every infoset value as one row of the output matrix, so an
infoset costs 500-1536 MAC. Our config set is variable — median 22, p99 567 —
so we cannot table those rows. The config encoder **generates** them instead and
the readout is a dot product. That is the only structural novelty, and it is the
standard open-vocabulary output layer.

`join_p(P)` does not move between iterations, so it is projected once per leaf
(`Net::join_cache`); that is what pays for a wide board vector. Most of the
network's multiply-accumulates sit in the trunk, which runs once per physical
leaf, and the rest on the per-iteration path. The blob is `641,505` weights;
the training-only ownership head adds `291` more.

```text
TYPE    64   coin-type token width    D        256  board vector, readout width
C       96   hex channel width        POOL     64   pooled config embedding
BLOCKS  8    trunk residual blocks    CFGH     128  config encoder hidden width
JW      128  join width               JBLOCKS  3    join residual blocks
JOIN_IN = 2 * POOL + 1 = 129  both beliefs and the seat, per iteration
MODEL_TAG = [5]               the only accepted `dims`
```

### The card describer

Nothing in the network names a *unit*. Each of the ten coin types in play — five
per player — is summarised by its `CARD_FEATS = 25` rulebook facts, and
everything that refers to a card refers to a row of that table:

```text
  card[t] = card2(gelu(card1(card_facts[t])))                   [NTYPE, TYPE]
  tok[t]  = card[t] + pile(pile_counts[t]) + seat[t / NSLOT]
```

There is no unit-identity embedding, so a draft the network has never seen is
describable rather than an unknown code, and
`card_features_separate_every_draftable_unit` pins the precondition — if two
cards shared a fact vector the describer would merge them silently. The card
table is fixed for a game, so `Net::cards` builds it for the two canonical views
once per solve and the trunk and the config encoder both index those rows.

**A stored row holds one-hots, not embeddings.** `card` is learned, so a replay
row that contained it would carry whichever weights were live when it was
written, go stale as training moved them, and pass no gradient back to the
describer. The row keeps raw facts and the network does the lookup, which also
keeps `write_public_features` a pure function of the position.

**Slot permutation is exact.** Which slot the draft gave a unit is a pure
relabelling, so every value must be invariant to permuting each player's five
slots. Nothing is read through a per-slot dense layer: the ten types are a set
of tokens, the hexes name their occupant with a one-hot, and the belief reaches
the join only through sums over configs. Measured permutation error is `7.5e-8`
against a value spread of `4.2e-2` — exact to float32
(`test_parity.py::slot_invariance`).

### The trunk

```text
  x[h] = hex_stem(hex_facts[h]) + tok_stem(tok[occupant(h)])
         + pos[h] + glob_stem(loose)         occupant term is 0 on an empty hex
  repeat BLOCKS times:
      a = gelu(ln1(x))                                          [N_HEXES, C]
      y = mix( a[h] ++ sum over neighbours(h) of a[n] )
      y += pool( mean_h(a) ++ max_h(a) )              broadcast over all hexes
      x = x + out(gelu(ln2(y)))
  x = gelu(ln_trunk(x))
  P = board_out( mean_h(x) ++ max_h(x) ++ loose )               [D]
```

KataGo-shaped: pre-activation residual blocks over the board's own hex
adjacency (`board().neighbors`), 37 tokens, no padding and no convolution to
fake the geometry with. The global-pooling bias in every block is KataGo's
second-largest measured ablation (`1.60x`) and costs 1.8% of a block; without
it a hex cannot learn anything about the far side of the board in fewer hops
than the board is wide.

### The config encoder

The value of a leaf is a counterfactual value **per information state**, so it
is indexed by the config: `v(PBS, c) -> scalar`. The encoder turns a config into
two vectors — the row the readout dots against, and the vector the belief pools.

```text
  u(c) = gelu(ln_cfg( sum over slots k of
                      gelu(cfg1( [hand_k, fd_k, bag_k] ++ card[k] )) ))
  f(c) = cfg_f(u(c))                                            [D]
  g(c) = cfg_g(u(c)) + sum_k sum_zone count[c][k][zone] * cfg_m(card[k])[zone]
```

`k` runs over the owner's five slots; a config's owner is the canonical query
whose card view puts that player's coins in the first `NSLOT` rows. The sum over
slots rectifies *before* it sums. A sum of raw linear maps is a linear map of
the sum, and the sum of the inputs has forgotten which count belongs to which
card — the one thing the encoder exists to remember.

`g` also has a linear path, and it is there because pooling happens *after* it:
`sum_c beta(c) g(c)` therefore carries the belief's exact expected holding of
every card, bound to that card. "They almost certainly cannot play an Archer
this turn" arrives as a marginal instead of as an average of GELUs. `cfg_m`
depends only on the card table, so it costs two rows per solve and fifteen
accumulations per config. The alternative — handing the join the belief's raw
count marginals — is what broke slot equivariance while v5 was being built:
`2.1e-2` of permutation error against a `4.2e-2` value spread, half the signal,
through a dense layer with per-slot weights.

`config_features_separate_every_config` pins that two distinct private states
never share a feature vector.

### The join and the readout

A canonical query is `q = 2 * row + player`; the same physical row seen by the
other seat is `q ^ 1`.

```text
  pooled[q] = sum_c beta_q(c) g(c)              over q's own belief support
  z    = jp[q] + join_b( pooled[q] ++ pooled[q ^ 1] )   jp = join_p(P), cached
  repeat JBLOCKS times: z = z + join_w(gelu(ln_join(z)))
  h    = ln_h( P[q] + join_out(gelu(ln_jout(z))) )    plain LayerNorm, no gelu
  v(c) = <f(c), h> + value_bias
```

Everything a belief does to a value happens in those `JOIN_IN = 129` numbers and
three 128-wide blocks. A dot-product readout has no output matrix to shrink, so
the small initialisation lands on the config side instead: `cfg_f` starts at
`std 1e-3` and every value starts at the bias.

This is the reference implementation's shape with its fixed-width private-state
dimensions replaced by learned functions, because War Chest's private states do
not fit in a table (hand x face-down runs to ~145k and varies by draft). The
belief is the same substitution on the input side.

The belief reaching the network as `sum_c beta(c) g(c)` is the one place a fixed
width is a real approximation: a belief is a distribution over a config space too
large to enumerate. It does not change the game being solved — CFR carries exact
beliefs internally — only the network's view of it.

### The flat blob

`NetLayout::new` is the definition and `Net.flat()` writes it. Linear matrices are
`[in, out]` row-major, embeddings `[n, width]`.

```text
  weights  card1, card2, pile, seat[2], hex_stem, tok_stem, pos[N_HEXES],
           glob_stem,
           (mix[2C -> C], pool[2C -> C], out[C -> C]) x BLOCKS,
           board_out[(2C + LOOSE) -> D],
           cfg1[(3 + TYPE) -> CFGH], cfg_f[CFGH -> D], cfg_g[CFGH -> POOL],
           cfg_m[TYPE -> 3 * POOL],
           join_p[D -> JW], join_b[JOIN_IN -> JW], join_w[JW -> JW] x JBLOCKS,
           join_out[JW -> D]
  biases   the same order, skipping the bias-free layers — pile, tok_stem,
           glob_stem, cfg_m, join_p and the two embeddings — then the scalar
           value_bias
  norms    (gamma, beta) pairs in application order: (ln1, ln2) x BLOCKS,
           ln_trunk(C), ln_cfg(CFGH), ln_join(JW) x JBLOCKS, ln_jout(JW),
           ln_h(D)
```

That order is the contract between torch, Rust and CUDA. A transposed matrix, a
bias attached to the wrong layer or a LayerNorm applied out of turn shows up in
`test_parity.py` and nowhere else until a training run has quietly learned
nothing.

### The auxiliary head

Training only: per **location** hex, a 3-way logit over who owns that location
when the game ends — player 0, player 1, neither. It reads the trunk's per-hex
output directly and enters the loss at `aux_weight = 0.15`. Ownership is
KataGo's largest measured ablation (`1.65x` with score) and it is a genuine
decomposition of the outcome here: you win by getting all six markers down. It
is deliberately absent from `flat()`, so the engine never evaluates it and it
costs nothing at inference.

### What is cached

Inside a solve only the beliefs move. The card table runs twice; the trunk and
`join_p(P)` run once per leaf; `f(c)` and `g(c)` run once per *distinct* config
(`Solver::intern_config`, and the trainer deduplicates a batch the same way).
What remains per iteration is the belief pooling, one
`[rows, JOIN_IN] x [JOIN_IN, JW]` matmul, three 128-wide residual blocks, one
`[rows, JW] x [JW, D]` matmul, a LayerNorm, and one `D`-long dot product per
config.

### Features

`rebel.rs::write_public_features` is public by construction: per-hex
occupancy/height/marker, per-player reserve/face-up/supply/eliminated counts,
slot identities, card properties, marker and count scalars, `plies_remaining`.
Bag, hand and face-down appear only through their public sum and their public
sizes.

Loss is a Huber (`beta = 0.5`) averaged over the configs in a query's belief
support and then over queries, plus `aux_weight` times the ownership head's
cross-entropy. Targets are clipped to ±1. `train.py::losses` also reports
`max |v_0 + v_1|` over the belief-weighted expected value of the two seat views
of a row: nothing forces the network to be antisymmetric, so that residual
measures whether it has learned to be.

The board's 180-degree symmetry is **canonicalisation, not augmentation**.
Rotating the board maps white's two starting locations exactly onto black's and
permutes the six neutral ones, so rotating and swapping the seats is an exact
symmetry of War Chest. That rotation is how the second seat view is produced:
every physical row becomes exactly two network queries, `2 * row` as stored and
`2 * row + 1` mirrored, and the network only ever reads a position from the
point of view of the player whose coins sit in the first `NSLOT` slots. The
solver does it in `Solver::encode` through `State::mirror`; the trainer does it
in `train/mirror.py` on the packed row, carrying the auxiliary target along with
its ten location bytes permuted and their owners swapped. There is no flag and
no choice: a row without its mirror is not something the network can be asked
about. `mirror.py::check_against_engine` pins the row transform against
`State::mirror` itself.

## 6. Training

```bash
python train/train.py out=mine minutes=30
```

Knobs are `key=value` and every one is a field of `train/config.py::Cfg`; an
unknown name is an error rather than a silently ignored flag.

**Phase 1 — greedy warm start** (`warm_minutes`, default 5). Both players are a
stochastic one-ply greedy bot on a public-information evaluation, and the value
target is that evaluation squashed into (-1, 1), blended with the realised game
outcome by `eval_mix` — default `1.0`, meaning pure evaluation. That makes the
warm label a deterministic function of the saved public state, which is the
whole point: at `eval_mix=0.5` half of every target was one sampled outcome
that no public state could explain, and the network fit the noise and entered
ReBeL with a broken value scale. The entire network trains on it; there is no
public-only phase and nothing is zeroed. The network at the end of this phase
is snapshot 0, labelled `init`.

**Phase 2 — ReBeL.** Self-play with a CFR solve at every decision. Defaults are
`depth=1`, `iters=64`, `cfr=dcfr`; the GPU golden run uses `depth=2`.

Drafts are randomised by default (`random_draft=1`); set it to 0 for the
rulebook's starter matchup.

**The replay sampler.** `recent_mix` of a batch is drawn from the newest
`recent_frac` of the buffer and the rest uniformly from all of it, which at the
defaults (0.5, 0.2) draws a fresh row six times as often as an old one while
leaving every row reachable. Old rows carry targets written by a network that has
since moved; the per-epoch `old=`/`new=` columns report the gap.

**Evaluation** is one round robin at the end, `tools/arena.py`: every snapshot
is archived as a bot and plays every other, and Greedy, fitted with
Bradley-Terry into an Elo each. Matches are paired — the same draft for both
seatings — and use a full solve and the reference strategy. Because a bot
carries the binary that can play it, snapshots from revisions that no longer
share a value network can be rated on one scale. See `docs/ARENA.md`.

Nothing is compared, promoted or selected while a run is going.

## 7. Tests

* `rebel_pbs.rs::features_do_not_leak_private_information` — swapping a player's
  true config for any other consistent with the same public counts must not move
  a feature.
* `rebel_pbs.rs::a_solve_reads_only_the_beliefs` — solve the same public position
  in two different worlds; root values and root strategy must agree bit for bit.
* `rebel_pbs.rs::config_features_separate_every_config` — two distinct private
  states never share a feature vector.
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
* `train/test_parity.py` — the Rust network against PyTorch on the same weights:
  worst relative error `9.6e-7` over six batch shapes, and the slot-permutation
  invariance both of them owe the draft.
* `scenarios.rs` (36 cases) and `invariants.rs` — the engine itself.

The Rust tests cover the engine, the PBS, the solver and the CPU network.

## 8. Tools

* `train/diagnose.py` — how learnable a dump's targets are, model-free.
* `train/dump.py` — reading a dumped replay buffer as a supervised dataset.
* `train/save_roots.py` — freeze a root sample from real self-play, for sizing.
* `examples/featstats.rs` — the real range of every feature.
* `examples/cfgvalue.rs` — how far the value separates configs.
* `tools/arena.py` — the ladder: archived bots, a referee, and Elo. It rates
  checkpoints from incompatible revisions on one
  scale.
* `train/export_weights.py` — the flat blob the Rust tools load.

## 9. Layout

```
engine/src/rebel.rs     PBS: configs, beliefs, observations, features
engine/src/search.rs    growing-tree CFR subgame solver
engine/src/selfplay.rs  self-play loop, belief filter, data collection, greedy bot, eval
engine/src/net.rs       the value network (Accelerate BLAS)
engine/src/py.rs        pyo3: set_weights / gen_data
engine/src/arena.rs     the referee and the bot protocol
engine/src/bot.rs       one bot's side of an arena game
engine/src/policy.rs    a node's policy, and the belief filter over it
train/value_net.py      the network in torch, and the blob both sides agree on
train/mirror.py         the 180-degree rotation that makes the second seat view
train/gpu_batch.py      replay rows -> canonical query batch on the device
train/train.py          PyTorch training loop, snapshots on a timer
tools/arena.py          the ladder: bots, referee, Elo
tools/monitor.py        the panels a run is read from, served live from runs/
```
