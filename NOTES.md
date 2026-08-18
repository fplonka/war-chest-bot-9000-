# The arena, and the four architectures on one ladder

Evaluation is now one system: a bot is a directory holding a frozen binary, its
weights and a manifest; a referee owns the true game and speaks to two of them
over pipes in public information only. `docs/ARENA.md` describes it. What it
replaced was three overlapping paths — a snapshot round robin, a two-process
cross-revision relay, and a one-off script — which could not be compared with
each other. The browser UI is the same referee with a person in one seat, so
the agent you play is the agent the ladder rated, down to the binary.

## The ladder

200 games per pairing, colour-swapped pairs over random drafts from the whole
pool, each bot at the search its own checkpoint trained with.

| bot | trained | Elo (greedy = 0) | 95% | score |
|---|---:|---:|---:|---:|
| `v5-2h` | 125 min | `1313` | `36` | `0.833` |
| `v4-12h` | 725 min | `1265` | `34` | `0.782` |
| `v4-2h` | 125 min | `1060` | `34` | `0.556` |
| `v3-trav` | 29 min | `817` | `45` | `0.328` |
| `greedy` | — | `0` | `217` | `0.001` |

The pairings, with the sign test over colour-swapped pairs:

| pairing | W-L-D | score | pairs | p |
|---|---:|---:|---:|---:|
| `v5-2h` over `v4-12h` | `111-88-1` | `0.557` | `30-18` | `0.11` |
| `v5-2h` over `v4-2h` | `161-38-1` | `0.807` | `66-2` | `1.6e-17` |
| `v4-12h` over `v4-2h` | `155-45-0` | `0.775` | `57-2` | `6.1e-15` |
| `v4-2h` over `v3-trav` | `161-38-1` | `0.807` | `66-3` | `1.9e-16` |
| `v5-2h` over `v3-trav` | `193-6-1` | `0.967` | `94-0` | `1e-28` |

**v5 at two hours is level with or ahead of v4 at twelve, but the gap is not
resolved at 200 games.** An earlier run of the same match came out `130-69`
(`p = 1.5e-4`) and this one `111-88` (`p = 0.11`); the seeding changed between
them, so they are two honest samples of a difference that sits near what 200
games can see. Everything else on the ladder separates cleanly. Take the
ordering; do not quote the v5/v4-12h margin without more games.

The `v5-2h` against `v4-2h` score of `0.807` is the same measurement the
retired cross-engine relay put at `0.8225`, under a different belief model and
a different protocol. Two independent implementations landing that close is the
reason to believe either.

Greedy loses 800 games to 1 across the field, so it anchors the scale and
nothing else. The pairing table is what to read.

## The tablebase, and two wrong turns before it

There is now an objective benchmark: positions where one side wins **whatever
the other does**, proven over the rules with no value network anywhere. A bot
plays the winning seat, makes one move, and the move is right exactly when the
win is still forced afterwards. No opponent, no sampling, no network — the same
questions and the same marking for any bot ever built.

Generation is nearly free: 3,000 games cost 182 CPU-seconds, so ten hours on
the box would be forty million games. It is not the constraint and never was.
Scoring is one solve per question — the position is shipped rather than
replayed into — so that is not the constraint either. What bounds a useful set
is statistics: a binary outcome at `n = 400` per bucket is `±5%` at 95%, and
past that the extra questions buy nothing.

So the set is **stratified**: a quota per bucket. Left alone the set fills with
easy positions, which are far more common than hard ones.

### Positions come from bots playing, not from a hard-coded policy

The generator watches two *bots* play and asks the referee, at each late
position, whether the result is already decided. That the players are bots is
the point: a bot is a frozen binary behind a protocol, so an architecture
change cannot break the generator — which matters, because the whole purpose of
the set is to survive architecture changes. It also makes the source of
positions a choice rather than a policy baked into a tool:

* `bots/random` on both sides gives positions from flailing. Cheap — six games
  a second on the box, so every quota fills in minutes — and mildly *out of
  distribution* for a net trained on self-play, which makes the set a
  generalisation test as much as a tactical one.
* A trained pair gives positions a real game reaches. Slower per game, and
  strong players leave fewer wins lying around, which is exactly why those
  positions are worth more.

The proof quantifies over the **whole opponent range**, not the hand they
happen to hold. A plan that only beats the actual hand is not forced, because
the winner cannot see it. This was briefly lost when generation moved from a
standalone binary into the arena, and it costs about five per cent of the
questions when it is put back — small enough to be easy to miss and large
enough to matter.

### A serialisation bug that had been silently corrupting positions

Found by the tablebase, and it predates it. `HexSet` is one bit per hex over
thirty-seven hexes, so a `u64` — and `roots.rs` wrote it as a `u32`. A position
saved at a `FootmanManeuver` came back with every hex from thirty-two up
missing, which turned a node with four legal moves into a node with none.
Nothing caught it because a truncated set is still a valid set, and the only
prior consumer of `roots.rs` was GPU tree sizing, where a slightly wrong tree
is a slightly wrong number rather than a crash.

It surfaced here because a question that round-trips into an unanswerable
position is a hard failure. Both directions now have a round-trip test with a
hex above thirty-two in it, and `ROOTS_VERSION` is bumped.

### The ply count is not a difficulty scale

The first real result, and it was not what was expected. Greedy, by plies to
win:

| plies | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| kept the win | `0.13` | `0.24` | `0.24` | `0.39` | `0.44` | `0.41` | `0.58` |

Deeper mates are *easier*, not harder. A win eight plies out usually has
several moves that hold it; a win in two often has exactly one. Mate in one is
free — take the marker — and is excluded from the set entirely, because every
bot answers it.

So the honest difficulty axis is **how many of the legal moves keep the win**,
which the generator measures directly:

| winning moves | 1 | 2 | 3 | 4 or more |
|---|---:|---:|---:|---:|
| kept the win | `0.21` | `0.38` | `0.29` | `0.73` |

and the set is stratified on that. The other axis that matters is whether any
hidden information is left: greedy keeps `0.43` of the wins where the
opponent's hand is already pinned down and `0.19` where it is not. A benchmark
that did not separate those two would have hidden its most interesting number.

### The two wrong turns

**Searching the whole game.** The first plan was endgames brute-forced to
terminal. The public tree multiplies by about twenty per ply:

| plies | nodes |
|---|---:|
| 1 | ~20 |
| 2 | ~500 |
| 3 | ~10,000 |
| 4 | ~150,000 |
| 6 | > 400,000 |

and a game runs for hundreds of plies, so that is `20^16` and change. True, and
it led to the wrong conclusion, because the measurement was taken on *positions
sampled from full-draft play* and then generalised to the game. A tablebase is
never sampled — it is constructed, or in this case *selected*: the question is
not "can this position be searched to the end" but "is this position already
decided", and those are very different sizes.

**Deep search as a stand-in for truth.** The fallback was to search each
position deeper than a bot can afford and score the bot's move against that. It
was built, run, and deleted. A deeper search puts the same value network at its
leaves, so it inherits that network's errors: it measures whether a bot agrees
with itself given more time. And it has to be built with *some* net, which
biases it towards that architecture — precisely the comparison it existed to
make neutral. A benchmark that flatters one family is worse than none.

### What the tablebase does not measure

The tail of a game, and only positions that happen to be forced. A bot can
convert every won endgame and still play a poor opening. It is a floor and a
check, not a verdict; read it beside the ladder.

## What a ladder costs

Two 3090s, `0.88` games/sec: a five-bot round robin at 200 games a pairing is
`2,000` games in `38` minutes. Two things got it there. The request and reply
streams were decoupled, so neither side waits for the other's slowest game.
And the referee stopped re-asking a bot to model the *same* position over and
over while its opponent thought — a bot models its opponent by solving their
node, and that solve was being repeated on every poll. That one fix took a
CPU-only ladder from about `1,000` to `8,500` games a minute.

# Value v5 — the architecture rebuild

## What was wrong with v4

v4 spent its compute in the wrong place. CFR re-asks every leaf on every
iteration: at depth two, `T=64` gives about 2,030 physical leaf rows and
158,000 belief-conditioned row-passes. v4 flattened the 37 hexes into a
three-layer 384-wide MLP, then put two more 384-wide layers on the repeated
path. It also assigned dense weights to draft-slot positions, although those
positions are arbitrary.

| | v4 | v5 |
|---|---:|---:|
| parameters | `1,550,177` | `641,796` (`641,505` in the weight blob) |
| public board | flat, `3 x 384` | 37 tokens, `8 x C96` residual GNN |
| paired player views | two public trunks | one shared physical trunk |
| repeated join | `2 x 384`, then per-config MLP | `3 x 128`, then dot product |
| depth-2 balanced generation | `574.6/s` mature run | `680.8/s` six-minute smoke |

DeepStack, ReBeL, TurboReBeL and Student of Games all avoid this by making the
infoset an **output index**: one tower per public leaf emits every infoset
value as one row of the output matrix, so an infoset costs 500-1536 MAC. v4
charged 33,000 MAC of GEMM plus a GELU per config. We cannot table the output
rows — the config set is variable, median 22 and p99 567 — so v5 generates
them from a config encoder and reads out with a dot product. This keeps the
output-index economics without assuming a fixed support.

## The shape

```
physical state ─► TRUNK (8 hex residual blocks, global pooling) ─► P   once/leaf
config c ───────► CONFIG ENCODER ─► f(c) readout, g(c) pooling      once/config
 [Σβ_own g, Σβ_opp g, seat] ─► JOIN (3 blocks, 128 wide) ─► h       every iteration
                             v(c) = <f(c), h> + bias
```

Widths: `TYPE=64, C=96, BLOCKS=8, D=256, POOL=64, CFGH=128, JW=128,
JBLOCKS=3`. The trunk is KataGo-shaped: pre-activation residual blocks over
the board's own hex adjacency, with a global-pooling bias in every block.
`join_p(P)` is cached once per physical leaf, which keeps the repeated path
small while the board vector stays wide.

### The central bet to test

Unlike DeepStack, ReBeL and Student of Games, v5 keeps the trunk public-only
and injects both beliefs in the join. This is deliberate: at depth two the
public trunk runs about 2,030 times per solve while the belief-conditioned path
runs about 158,000 times. It is also an expressivity tradeoff. The hex features
cannot be recomputed after learning that the opponent probably lacks a given
coin. Treat this split as an experiment, not settled literature. The first
architecture ablation should move one narrow hex block after belief injection
and measure strength at matched solver throughput.

## Slot-permutation equivariance, and the bug it caught

Which slot index the draft assigns a unit is a pure relabelling, so the network
must be exactly invariant to permuting each player's five slots. v4 was not:
`value_net.py:100` flattened the ten `[pile counts ‖ token]` blocks in slot
order into a dense layer, so the same unit got different weights depending on
the draft. That partly undid the card describer built to avoid exactly this.

v5 treats the ten coin types as a set of tokens described by their printed card
facts, with no unit-identity embedding at all. The first draft still
failed the check — measured `2.1e-2` against a value spread of `4.2e-2`, half
the signal — because the belief's raw count marginals were fed to the join as
30 numbers through a dense layer with per-slot weights.

The fix removes them and makes the pooling vector carry them instead:

    g(c) = cfg_g(u(c)) + Σ_k Σ_{zone} count[c][k][zone] · V[k][zone]

with `V[k] = cfg_m(card_k)` a per-zone embedding of the *card*. `cfg_m` is
linear in the counts, so `Σ_c β(c) g(c)` carries the belief's exact expected
holding of every card, bound to that card — "they almost certainly cannot play
an Archer this turn" arrives as a marginal rather than as an average of GELUs.
`V` depends only on the card table, so it costs two rows per solve and fifteen
accumulations per config. The join input shrank from 158 to 128 in the process.

With that, permutation error is `7.5e-8` against a `4.2e-2` spread — exact to
float32. TurboReBeL's largest architecture-free win was its `24x` isomorphic
augmentation; this is the same kind of symmetry, taken by construction instead
of by augmentation.

## Auxiliary head

Per location hex, a 3-way logit over the final owner of that location at the
end of the game the row came from. This is Go ownership, which is KataGo's
largest measured ablation (`1.65x` with score), and it is a genuine
decomposition of the outcome here: you win by getting all six markers down.
Training only — it is not in the weight blob and the engine never runs it.

**It is dead in the ReBeL phase, and the logs say so plainly.** Through the
Greedy warm-up the head learns: cross-entropy `0.90 -> 0.45`, accuracy
`0.62 -> 0.80`. Ten seconds into ReBeL it reads `0.029/0.986`, and from the next
epoch to the end of a thirty-minute run it reads `0.00013` at accuracy exactly
`1.000`, every epoch.

The cause is where the label comes from. `selfplay.rs::push_row` writes the
ownership *at the solve site* and `backfill_owners` replaces it with the
finished game's ownership only if that game ends before its rows are detached.
Greedy warm-up generates whole games, so its labels are the real end state and
the target is a real prediction. A streaming depth-two ReBeL game takes minutes,
so its rows are detached mid-game and keep the solve-site label — and each
location's current marker owner is a *raw input feature* of the row the head
reads. The head learns the identity map, and `aux_weight = 0.15` of the loss
buys nothing for the whole run that matters.

The cheap fix is a per-row bit set by `backfill_owners`, with the auxiliary loss
masked to the rows that carry it; the honest fix is to write the label back into
the replay buffer when the game finishes. Either way the KataGo justification
does not currently apply to the ReBeL phase, so treat the head as untested
rather than as an established win.

## Trained-weight audit

The twelve-hour v4 checkpoint has no dead input group or frozen layer. The
first public matrix gives comparable RMS weight to hex facts (`0.186`) and unit
tokens (`0.188`); the repeated context gives comparable weight to public state
(`0.160`), own belief (`0.166`) and opponent belief (`0.148`). Its public
matrices still have effective ranks `337`, `240`, `195`, and every major matrix
moved another `0.006-0.033` RMS in the final two hours. The plateau is
therefore not a dropped input, rank collapse or stopped optimiser. It is the
cost and sample inefficiency of relearning board geometry through a flat map,
while bootstrapped targets keep moving.

After four ReBeL minutes, every v5 spatial block has moved from its warm
checkpoint. Their effective ranks are `74-86` of 96; all three join blocks are
`98-99` of 128. The depth is active rather than decorative. The same smoke
sustained `680.8` balanced depth-two solves/s with no dropped or exclusive
work; CPU/Torch blob parity is `1.83e-6` relative and slot permutation changes
values by at most `6.68e-6`.

## The depth-two baseline, and what it beats

`v5_d2_125m`: 125 minutes from scratch at seed 95, five Greedy minutes,
snapshots every 25 so the checkpoints land on the two anchors worth playing.
`2,657,646` solves, `57,928` games, `369.2` balanced solves/s over the whole
run, three oversize routes and no dropped work. It learns monotonically by
every internal measure: target SD `0.083 -> 0.648` across the six checkpoints
while loss over target variance falls `1.68 -> 0.031`.

Its own ladder, over all six checkpoints with Greedy pinned at zero, is
monotone and had not flattened: `-50`, `+371`, `+638`, `+692`, `+721`, `+743`.
Ladder Elo cannot rank close policies -- this file has found that twice -- but
it can say a run kept learning, and this one did.

Against the two architectures it replaces, on the cross-engine relay at matched
search (depth 2, `T=64`, colour-swapped pairs, every state cross-checked
between the two builds):

| new | old | W-L-D | score | Elo | paired | paired p |
|---|---|---:|---:|---:|---:|---:|
| `v5 @ 30min` | `traverser` final (v3, 29min) | `157-42-1` | `0.7875` | `+227.6` | `62-4-34` | `2.1e-14` |
| `v5 @ 125min` | `d2t64_long12h.s1` (v4, 125min) | `164-35-1` | `0.8225` | `+266.4` | `69-4-27` | `2.4e-16` |

Both are like-for-like on wall clock: the same minutes of training on the same
two cards, each side running its own engine revision. v4's own ladder put that
`s1` checkpoint at `+501` against Greedy, which is the scale these Elo numbers
sit on.

The weights say where the late learning happens. From 30 to 125 minutes the
config encoder and the join move most -- `cfg_f` by `145%` of its own RMS,
`join_b` `64%`, `cfg_g` `61%`, the join blocks `56%` -- while the trunk's stems
move `18%` and the position embedding `5%`. Board geometry settles early; the
belief-conditioned path is what keeps improving. Read that beside the device
profile below, where the repeated path is already the larger cost, before
deciding which side to widen.

## Throughput

The depth-two baseline, thirty minutes from scratch at seed 95 with a five-minute
Greedy warm-up, sustains **`543.6` balanced solves/s**: `815,276` solves,
`16,556` games, no dropped work, no exclusive route, `debt` `688` of `3.26M`
optimizer rows. A mature six-minute stream from that final checkpoint with the
horizon payoff at zero runs at `453.3/s`, which is the number to A/B against.
The rate falls with run length rather than with anything in the code: a
125-minute run averages `369/s`, because mature games sit in midgame states
whose belief supports are three times the opening's.

Both cards are `90-92%` busy with a mean resident-thread occupancy near `100%`,
so this is device-bound, not host-bound. Where the device time goes, over 45
seconds of a mature stream:

| | share |
|---|---:|
| `trunk_row` (the whole board trunk) | `30.3%` |
| join GEMMs (cuBLAS, tensor cores) | `~16%` |
| `join_block` / `join_finish` / `join_input` | `9.3 / 5.4 / 4.3%` |
| `readout` | `8.1%` |
| `reach_sweep` / `backprop_sweep` | `6.5 / 5.3%` |
| `belief_sums` | `4.2%` |

The per-iteration join path is therefore the larger half at about `46%`, and a
third of the device's time sits in five kernels that only move `128`-wide rows
in and out of the arena.

### What paid, and what is left

The join blocks' bias moved into the matrix the GEMM already applies, as its
`JW + 1`-th row against a constant `1` in the pre-activation: `471.5` against
`453.3 solves/s` on the matched stream, `+4.0%`, which is the second write of
the residual stream per block disappearing. The same trick fits `join_input`
and `join_finish` and is worth about as much again.

What is left after that is the two sweeps at `11.8%`, which walk their
reverse-gather rows one thread per row and so read uncoalesced, and the readout
at `8.1%`, which re-reads `f(c)` per leaf and per config where the interned pool
would allow one batched GEMM per support. Beyond those, `600/s` at depth two
wants the network cheaper rather than the kernels tighter -- the trunk does
`8.3` MMAC per leaf row against v3's `0.8`, which is exactly why v3 ran at
`1200/s`.

### The fused trunk, measured and reverted

The eight residual blocks used to run as six kernels each, moving about `1.4 MB`
per row through the arena. One kernel per row, with the residual stream in
registers and the normalised hexes and their neighbour sums in shared memory,
does the same arithmetic with no DRAM traffic at all: it cut the build stage
from `277` to `154 ms` on a `57k`-row wave and passed every oracle, including
`full_wave_oracle` and the exact-math head comparison.

It is nevertheless *slower* end to end -- `429.5` against `453.3 solves/s` on
the matched six-minute stream -- and the reason is the two shared buffers. At
`33 KB` per block only three blocks fit an SM, which is `50%` thread residency,
and they leave nothing for the other three lanes of the same device, so the
saved arithmetic never reaches the stream. Reverted. The obvious next attempt is
to hold both buffers as halves: that is what the old path fed the mix GEMM
anyway, and `17 KB` per block would let the kernel share an SM.

## Deliberately not done

* **Zero-sum enforcement.** Three v4 attempts lost. `odd` showed the
  reparameterised readout learns fine (residual `0.0006`, ladder `+578`) and
  died on throughput (`844` against `1081` solves/s). The game is zero-sum; a
  correct architecture should learn that, and forcing it has only ever cost
  strength here. Left as a diagnostic.
* **Policy head.** The v4 measurement (`arch_policy`, `646` against `663` and
  `696`) tested it as an auxiliary loss only; the warm-start run that was its
  actual justification was killed before producing a number. Worth revisiting
  as a search change, not as a loss term.
* **Direction-aware message passing.** The trunk sums its six neighbours with
  one shared weight, so it sees adjacency and distance but not *heading*.
  Crossbowman and Lancer need a straight line, and lines of advance matter
  generally. Per-direction taps cost `2.7x` the trunk; three axis pairs cost
  `1.67x` and would still express a line. This is the first trunk experiment to
  run once the box is free.


# Value v4 runs

## Main finding

The failed warm starts mixed two incompatible labels. With `eval_mix=0.5`, each Greedy target was half deterministic static evaluation and half one sampled game outcome. During `public_only` warm-up the network had no private information that could explain the sampled outcome. It therefore fit high-variance label noise and entered ReBeL with a poor value scale. `eval_mix=1.0` makes the warm label a deterministic function of the saved public state. The same architecture then fit a fresh matched-generator holdout (`RMS 0.0081`, correlation `0.9998`) and trained successfully.

This setting affects Greedy warm-up only. ReBeL continues to use its bootstrapped counterfactual targets.

## Runs

| Run | Change | Result |
|---|---|---|
| `value_v4_beliefw` | Belief-weighted loss; 10 min, 3 min warm, 4 optimizer rows/solve | No clear improvement over the broken baseline. Final target SD `0.193`; balanced generation `228 solves/s`. |
| `value_v4_public_warm3` | Public-only warm-up with private paths zeroed; `eval_mix=0.5` | Training loss fell, but fresh-state behavior remained poor because the warm labels still contained sampled-outcome noise. |
| `value_v4_public_warm30` | Same, plus cached public projection | Stopped early. Cached projection improved the frozen-tape kernel benchmark from about `753` to `820 solves/s`, but the run still used the bad warm labels and too few optimizer updates. |
| `value_v4_public_warm30_r12` | 12 optimizer rows/solve | Stopped early after the initial instability. More fitting cannot repair an unlearnable warm target. |
| `value_v4_warm5_ratio1` | Five-minute warm-only control with `eval_mix=0.5` | Replay fit improved while a matched fresh-generator holdout remained poor (`RMS 0.509`, correlation `0.455`). This isolated label noise/data efficiency rather than optimizer capacity. |
| `value_v4_warm3_clean` | Three-minute warm-only control with `eval_mix=1.0` | Decisive control: matched holdout `RMS 0.0081`, correlation `0.9998`. |
| `value_v4_rebel_clean3b` | Start ReBeL from the clean warm checkpoint; 12 optimizer rows/solve | Stable three-minute bootstrap check. Target mean stayed small and loss fell to `0.00027`. |
| `value_v4_clean30` | Clean end-to-end run; 3 min deterministic public warm, then ReBeL; 12 optimizer rows/solve | Healthy learning. Ladder Elo versus Greedy: init `-120`, s1 `-94`, s2 `+83`, s3 `+268`, s4 `+393`, s5 `+425`, final `+431`. Final beat Greedy `19-0-1`; final and s5 were tied within noise (`39-35-6`). |
| `value_v4_depth1_precise30` | Production defaults: depth 1, 32 iterations, 4 optimizer rows/solve, and fused context normalization | Verified 30-minute run: `10,835,186` solves, `7,225` raw and balanced solves/s including drain, no dropped jobs, and no exact fallbacks. The ladder improved at every checkpoint: init `-91`, s1 `+396`, s2 `+499`, s3 `+524`, final `+562` Elo versus Greedy. |

## `value_v4_clean30` dynamics

The raw fresh-row loss spike was a curriculum wave, not divergence. At 16.4 min, `loss_new=0.0206` while `loss_old=0.00055`: newly generated targets had become much harder. About nine minutes later those rows had aged into the old replay bucket; `loss_old` peaked at `0.0109` while `loss_new` had fallen to `0.0045`. By 29.8 min the buckets converged at `0.0090/0.0082` while target SD had expanded to `0.278`. Total loss ended near `0.11` of target variance.

Most playing-strength improvement occurred from s2 through s4. The final two minutes added no measurable strength over s5. This motivates measuring generation throughput, then testing a simpler schedule over a longer ReBeL interval rather than changing several learning knobs at once.

## Performance controls

Caching the public first-context projection once per GPU wave improved the frozen-tape benchmark from `753` to `820 solves/s` (`+8.8%`). Replacing GELU with ReLU did not improve production throughput and was reverted. Reducing the training search from 64 to 32 iterations increased a controlled run from `553.5` to `653.3 solves/s` and from `436,269` to `466,622` rows in the same wall time. The 32-iteration setting therefore stays.

The optimizer ratio returns to 4. The legacy successful run used 4, while 12 trained `7.83M` optimizer rows from only `4.48M` generated rows. A ten-minute ratio-4 continuation, `value_v4_ratio4_10`, generated `1.28M` rows and trained `0.75M` rows without material optimizer debt. Its final tied its input checkpoint `273-270-57` over 600 matched games. More replay fitting did not provide measurable strength.

The production configuration exceeds the `800 solves/s` requirement by `9.0x` over a full 30-minute run. It generated `75,136,466` replay rows from `10,835,186` solves, admitted only three oversize routes, and dropped no work. The independently evaluated checkpoints strengthen monotonically from 5 through 29 minutes; the final beat the 23-minute checkpoint with an aggregate score of `0.545` over 200 games.

## Depth-two, 64-iteration DCFR throughput

A deep solve is a different machine from a depth-one solve. At steady state a
depth-two subgame carries `2,181` leaf rows against roughly `180` for the
shallow default, and the head runs `78` times per solve: `64` CFR iterations
plus one fixed-policy pass per player for each of the seven carried root
beliefs that `snapshot_iters(64)` produces. That is `1.17e11` FLOP of head GEMM
per solve, so `800 solves/s` needs `93.7 TFLOP/s` sustained. Two RTX 3090s
offer `71.2 TFLOP/s` of FP32, which is why the FP32 path cannot reach the
target and the tensor cores are not optional.

Controlled four-minute runs, depth 2, 64 DCFR iterations, seed 101, 32 actors
per worker so games complete and the solve-size mixture settles:

| build | solves/s | solves | exclusive routes |
|---|---:|---:|---:|
| before | `294.9` | `70,761` | `44` |
| chunked public tower and kernel traffic | `328.0` | `78,716` | `0` |
| tensor-core head GEMMs | `531.6` | `127,565` | `0` |

The public tower is a per-row map, so it now runs in row chunks. Its input row
is `PUBLIC_IN` floats wide and sizing that buffer for a whole wave made a
mature wave reserve over four gibibytes, which routed it to the exclusive
one-job lane; chunking removed every such route and the redundant device copy
that used to cache the public context projection. LayerNorm now holds its row
in registers across the mean and variance passes instead of reloading it from
the arena twice, which brought `context_norm_gelu` and `norm_gelu` to `937
GB/s` -- DRAM peak, so they are done. The readout keeps the context row with
the joint bias folded in, and the output weights, in shared memory, reads
candidates as `float4`, and values four leaves per block so that neighbouring
leaves share their candidate pool through L1. The sweeps name their arena and
table bases once instead of reloading them through `w` after every store.

With the tensor-core GEMMs the mix is `40.4%` GEMM, `12.8%`
`context_norm_gelu`, `11.0%` readout, `9.8%` `norm_gelu`, `8.1%`
`belief_sums`, `7.6%` reach sweep, `6.7%` backprop sweep.

Tiling the readout exposed a race that predated it. `belief_sums` parks the
opponent reach mass in the value slot the readout is about to overwrite, and
the readout read it with no barrier between the warps that read and the warps
that write. Whether a leaf got the reach mass or a leaf value depended on warp
timing; the reuse-invariance check caught it intermittently, at up to `0.30`
absolute in a single action probability. The read now happens before the
block's one barrier.

Holding the candidate, belief and public-context embeddings as halves was tried
and reverted: `522.5` against `531.6 solves/s`. The readout is instruction-bound
on GELU, not byte-bound -- `tanhf` already compiles to one `tanh.approx.f32`, so
`96` of its `130` inner instructions are the nonlinearity -- and the extra
`__half22float2` conversions cost more than the halved loads saved. Only
`context_norm_gelu` was genuinely at DRAM peak, and it holds one of three
streams. Precision was not worth spending there.

The oracle contract splits along the same line. `Blas` carries the math mode, so
the three CPU-oracle tests build a precise executor and keep their original
tolerances against exact math, while `tensor_core_head_tracks_exact_math` runs
the production path and bounds it against that same exact solve: root values,
which are what training consumes, agree to `1.9e-4`, and a regret-matched action
probability -- a ratio of differences between leaf values, so two orders more
sensitive -- to `4.8e-2`.

The verified 30-minute run, `value_v4_d2i64dcfr_30`, sustained `522.2 solves/s`
at production defaults: `783,039` solves, `3,131,392` optimizer rows, `14,161`
games, no exclusive route, no dropped work, no exact fallback. DCFR trains well
at this depth -- ladder Elo against Greedy went init `-49`, s1 `+193`, s2 `+498`,
s3 `+638`, final `+681`, and the final beat Greedy `20-0-0` in its seeding pair.
That is `1.77x` the pre-optimization rate and `65%` of the `800 solves/s` target.

CUDA Graphs and larger waves were both tested and rejected: graph capture cannot
reuse an executable across the shapes a live stream produces, so it measured
`466.1` against `531.6 solves/s`. Reaching `800` from here needs the non-GEMM
half cut by about `2.4x`. The two places that can give it are the LayerNorms,
which are `22.6%` of GPU time and sit exactly at DRAM peak in FP32 -- half
storage halves them -- and the reach and backprop sweeps at `14.3%`, which walk
their reverse-gather rows one thread per row and so read uncoalesced.

## The from-scratch depth-two collapse

The bootstrap collapse was a recipe defect, not a search or estimator defect,
and the variable is the size of the live game set. Four seeds, seven minutes
each, five of them warm, everything else identical:

| seed | live 1152 (spread, games) | live 4608 (spread, games) |
|---|---|---|
| 95 | `0.097` -> `0.136`, 61-185 | `0.001`, 0 |
| 96 | `0.104` -> `0.165`, 94-263 | `0.049` -> `0.056`, 0 |
| 97 | `0.199` -> `0.241`, 131-245 | `0.001`, 0 |
| 98 | `0.102` -> `0.144`, 46-220 | `0.067` -> `0.077`, 0 |

Four of four train at `1152`; none of four train at `4608`, and not one of them
completes a single game in any ten-second window. That is the mechanism. A
depth-two solve costs about `550/s` against `7,900/s` at depth one, so with
`4608` games resident each one advances `0.12` decisions a second and a
hundred-decision game needs some `19` minutes to finish. Until a game finishes,
no terminal outcome has entered the replay buffer, so every label the network
trains on was produced by the network itself. `network -> CFR -> network` has no
shortage of fixed points when nothing anchors it, and any constant is one; the
runs at `0.049` and `0.067` are caught drifting monotonically toward theirs.

This also explains the seed lottery it was mistaken for. The `19` minutes are
comparable to a thirty-minute run, so a long run does eventually start finishing
games -- `value_v4_d2i64dcfr_30` finished `14,161` of them -- but only after its
target scale has already degenerated, which is why that run ends up *below* its
own initialisation (init `544`, six minutes `407`, thirty minutes `499`). Short
runs and short-lived seeds never get there at all. Sizing the live set removes
the lottery rather than improving the odds.

The fix is one knob instead of two. A worker now holds exactly as many live
games as it can have solves in flight (`gpu_inflight`), because a game held
beyond that can never be worked on sooner and only takes longer to finish;
`gpu_actors` is deleted. It is also faster, not a trade: `596 solves/s` against
the `390-522` the oversized set managed, since games that finish concentrate
beliefs and keep subsequent solves small.

Two hypotheses recorded in `TODO.md` were measured and both are refuted. The
value target is not the wrong estimator: we implement TurboReBeL, whose Phase 2
(Algorithm 2, line 13) specifies exactly `v^s(b) <- UpdateCFV(S', s)`, the
backpropagation under the fixed final reference strategy that `value_under`
performs, and that Phase 2 is what earns the `T+1` rows per decision
`selfplay_walk.rs` pins. The zero-reach uniform fallback never fires: over a
depth-two, 64-iteration solve under a flat network -- precisely the warm-start
condition it was suspected to spoil -- no reach among `114,585` rows is zero or
even below `1e-30`, and no strategy sum among `64,822` is zero. The reference's
`1e-80` floor is not representable in an `f32` arena in any case.

Half-precision head *activations* were then tried and reverted, and this one is
worth keeping written down. Holding Xb, H and H2 as halves -- feeding the tensor
cores their operands in the form they already multiply in -- is worth `+12.7%`,
`599.1` against `531.6 solves/s`, and it passes every oracle: root values still
track the exact CPU solve to `1.9e-4`. It nevertheless destroys the ReBeL
bootstrap. From a five-minute warm start the target spread collapses from
`0.091` to `0.001` within two ReBeL epochs and never recovers, at both seeds
tried, where the FP32 build at seed 95 does not. A warm network's within-support value spread is a few percent of its
across-query spread, so the differences CFR needs are near the half grid at the
start of the bootstrap, and quantising them away leaves the search nothing to
differentiate. Starting from an already-trained checkpoint hides this completely,
which is why the four-minute controls looked healthy. The lesson generalises: a
solve accuracy bound measured against a converged network says nothing about
whether the bootstrap can get there.

## The verified depth-two run

`d2t64_fixed_30`, thirty minutes from scratch at seed 95 with the live set
sized to inflight, is the first depth-two bootstrap here that learns
monotonically: `861,850` solves, `19,022` completed games, `horizon=0.01`, so
the games end in real outcomes rather than at the draw cap.

| checkpoint | trained | Elo vs Greedy |
|---|---:|---:|
| init | 5min | `-86` |
| s1 | 11min | `+373` |
| s2 | 17min | `+474` |
| s3 | 23min | `+563` |
| final | 29min | `+611` |

Against `value_v4_d2i64dcfr_30`, the old-recipe thirty-minute run at the same
seed, depth and iteration count, its final won `125-70-5` over 200 games --
score `0.637`, `+98` Elo, `z=3.9`, `p=1e-4`. Against the pre-refactor
`traverser` final on the cross-engine relay at matched search it won
`130-66-4`: score `0.66`, `+115.2` Elo, `p=5.7e-6` decisive and `1.9e-5` on
the colour-swapped pairs. Sustained throughput rose from `522.0` to
`574.6 solves/s`; the `850-988/s` of the early windows is not sustainable,
since trees grow as the policy sharpens.

What this does *not* show is depth two beating depth one at thirty minutes. It
does not: `value_v4_depth1_precise30` scores `+386.6` against the same
traverser where this run scores `+115.2`, because in the same wall-clock it
buys `10,835,186` solves against `861,850` -- a factor of `12.6`. The case for
depth two rests on the target quality per solve paying for that factor over a
run long enough to show a crossover, and thirty minutes is not that run. What
is settled is that depth two now trains at all, reliably, and that its own
curve no longer bends downward.

## Architecture comparison

Against the `traverser` final -- the last checkpoint of the pre-refactor
parameterised architecture, run on its own engine revision through the
cross-engine relay at matched search (depth 2, 64 iterations, colour-swapped
pairs, every state cross-checked between the two engines):

| new checkpoint | W-L-D | score | Elo | paired | paired p |
|---|---:|---:|---:|---:|---:|
| `value_v4_depth1_precise30.final` | `178-17-5` | `0.9025` | `+386.6` | `84-3-13` | `1.4e-21` |
| `value_v4_d2i64dcfr_final.final` | `177-18-5` | `0.8975` | `+376.9` | `82-0-18` | `4.1e-25` |

The first row is the fair comparison: both sides are thirty minutes from scratch
on their own revision. The second starts from the first, so it measures the deep
run's checkpoint, not thirty minutes of deep training from nothing -- and it is
slightly *behind* its own initialisation, which the run's own ladder also shows
(init `544`, six minutes `407`, thirty minutes `499`). Training on depth-two
targets moves the network off the depth-one optimum and `828k` solves is not
enough to get back.


The corrected full-network run, `value_v4_fullwarm30`, completed 30 minutes and generated `652,807` solves. In a 600-game direct match its final checkpoint beat Greedy `591-4-5`. The legacy `odd` final, evaluated with its own stable pre-refactor engine under the same seed and search settings, beat Greedy `567-3-30`. This anchor does not show an architecture regression.

Checkpoint-to-checkpoint results are strongly non-transitive. `value_v4_fullwarm30.final` tied its post-warm checkpoint `294-296-10`; `odd.final` also tied or slightly lost to `odd.init` (`262-274-64`) even though their Greedy results differed sharply. Elo chains and one opponent can therefore diagnose gross failure, but not rank close policies. The refactored readout is not the current blocker: deterministic full-network warm-up already produces a strong policy, and later self-play changes behavior without a stable matched-policy gain.

The direct cross-engine ladder removes the Greedy anchor. The current final beat
`odd.final` 349-218-33 over 600 games (score `0.6092`, `+77.1` Elo). The
colour-swapped pair test was 116-48-136 (`p=1.12e-7`, two-sided), and each of
three independent 200-game shards favored the current checkpoint. This rules
out an architecture regression in the trained stack. It does not isolate
architecture from training trajectory; that would require matched-data
training runs for both readouts.

## The train-to-generate ratio

Four twenty-minute runs at seed 95, depth two, 64 iterations, identical but for
`train_gen_ratio`, then their finals played directly against each other rather
than compared through separate Greedy-anchored ladders, which this file has
already found cannot rank close policies:

| ratio | solves | optimizer rows | balanced | games | direct score vs `4` |
|---|---:|---:|---:|---:|---:|
| 2 | `582,367` | `1,164,288` | `647.0/s` | `10,058` | `0.302` over 200 |
| **4** | `577,507` | `2,309,120` | `641.5/s` | `12,550` | -- |
| 8 | `491,968` | `3,935,232` | `546.8/s` | `7,818` | `0.150` over 20 |
| 16 | `367,542` | `5,879,808` | `408.4/s` | `6,312` | `0.050` over 20 |

Joint ratings put `4` at `731` against `2` at `587`, and `930` against `643` for
`16` and `541` for `8`. The margins are `p=2e-8`, `4e-5` and `2.6e-3`. Four wins,
and the reason it wins is legible in the two columns beside the result.

Generation saturates near `645 solves/s`. Ratios `2` and `4` both reach it --
`582,367` and `577,507` solves are the same number -- so dropping to `2` buys no
extra data and merely halves the passes taken over data already in hand. Above
`4` the throttle engages the other way: the optimizer holds generation back to
keep its ratio, and `16` pays `36%` of its solves for passes over rows the
network has already moved past. Since ReBeL's targets are induced by the current
network, re-fitting stale rows harder is chasing a distribution that has left.

So the right ratio is the largest one that does not throttle generation, and
that is identifiable without playing a single game: run at the ceiling and read
`debt` -- at `4` it is `908` rows against `2,309,120`, four hundredths of one
percent, meaning the optimizer keeps pace exactly. The criterion travels. If
kernels get faster or solves get deeper the ceiling moves, and the same
debt-at-ceiling reading will name the new ratio; `4` is the answer for depth two
at `T=64` on two 3090s, which is where the default already sat.

## Depth one against depth two, and what the Greedy anchor hides

At matched search -- both sides playing depth two, 64 iterations, through the
relay with colour-swapped pairs -- the depth-one thirty-minute run beats the
depth-two thirty-minute run `139-56-5`. That is `0.2925` to depth two,
`-153.4` Elo, `p=2.5e-9`, and `51-7-42` on the pairs. The two runs are
otherwise the same experiment: thirty minutes from scratch, seed 95, same
architecture, differing in the recipe they train under (`depth 1/T=32` against
`depth 2/T=64`).

The per-run ladders say the opposite. On its own Greedy-anchored ladder the
depth-two run finishes at `+610.7` and the depth-one run at `+561.8` -- depth
two ahead by `49` where direct play puts it behind by `153`. This is worse than
the imprecision already noted in this file: the anchor *misorders* them. Both
nets beat Greedy far too easily for the margin to carry information, so the
rating saturates and what is left is noise around a ceiling. Run reports and
ladder Elo can show that a run learned; they cannot choose between two recipes.
Every recipe decision here was settled by direct play for that reason.

Depth two was still improving at thirty minutes, so the question is whether it
crosses over later. Measured against a fixed yardstick -- the depth-one final,
matched search, 200 games each -- it does not look like it:

| depth-two checkpoint | trained | Elo vs the depth-one final |
|---|---:|---:|
| s2 | 17min | `-354.5` |
| s3 | 23min | `-217.3` |
| final | 29min | `-153.4` |

The gap closes by `+137.2` and then `+63.9` per six minutes. Reading the first
of those alone predicts a crossover at about `43` minutes; the second says the
approach is geometric, decaying by `0.466` every six minutes, `tau = 7.9min`,
with an asymptote `97.7` Elo *short* of the depth-one final -- and the depth-one
final is itself only a thirty-minute checkpoint that would keep moving. Three
points fitted across the tail of a bootstrap transient is a weak extrapolation
and does not prove an asymptote, but it does rule out the near-term crossover,
which is the claim that would have justified depth two for the long run.

So the recipe for a long run is depth one on this evidence, and the reason is
arithmetic rather than subtle: depth two buys better targets at `12.6x` the
price per target, and nothing measured here shows the quality repaying the
count. The case for depth two now needs a run long enough for depth one to
actually saturate against a fixed opponent, which thirty minutes is not.

## What made the endgame generator fast

Generating proven positions was, measured, 95% one thing: the forced-win
search. It ran once per position from Python, on one core, holding the
interpreter lock, and it was asked the same question about seven times over.

Four changes, in order of what they were worth.

**Return the distance, not a yes or no.** Asking "settled in two? in three? in
four?" ran a fresh search per depth. One search that returns the *least* number
of plies answers all of them, and a win found at some distance tightens the cap
on every branch still to be tried. The count of moves that keep the win falls
out of the same root scan, so sharpness stopped being a second search.

**Prove in bulk, inside the referee.** The referee now records every decision
node its games pass through, and the sweep drains that queue across every core
with the interpreter lock released. Driven a position at a time from Python
there were rarely more than three positions in hand and the cores sat idle.

**Cut the node budget.** This was the surprise. The budget never lets an
unproven position through — it only decides which are cheap enough to settle —
and the few positions it gives up on cost more than all the others together.
400,000 to 25,000 loses 1.5% of positions and runs 2.6 times faster, and the
losses are spread evenly over depth rather than concentrated in the deep ones.
Interpreted: the cost distribution has a tail so heavy that one position was
serialising a whole parallel sweep.

**Reuse the action buffer.** One vector per ply of depth instead of one per
node.

Together: 300 random-play games went from `65.3s` to `5.3s`, for 1.4% fewer
positions — `12.3x`. Under self-play the search is no longer the limit; the
cards are, which is where the cost belongs.

One thing that did *not* work: ordering moves by the control-marker swing, to
find refutations sooner. It made the run 1.8 times *slower*. The ordering needs
a full apply per child at every node, and that costs more than the cut saves.

## The proven-endgame benchmark, and what it says about five bots

A position is kept only if the side to move wins whatever the other side does,
within eight plies, *against every hand the opponent could be holding*. The
rules decide that, not a network, so the same 2,149 questions mean the same
thing to any bot ever built. 8,000 random-play games produced them in 75
seconds.

Bucketed by the share of legal moves that keep the win, which is the axis that
actually separates bots:

| moves that win | n | random | greedy | v4-2h | v5-2h | v4-12h |
|---|---:|---:|---:|---:|---:|---:|
| under 10% | 314 | 1.3% | 27.1% | 51.3% | 59.6% | 57.3% |
| 10-25% | 639 | 7.0% | 20.2% | 61.0% | 61.4% | 70.3% |
| 25-50% | 229 | 34.9% | 34.5% | 62.5% | 63.3% | 65.5% |
| over 50% | 967 | 95.1% | 95.9% | 94.0% | 92.9% | 93.0% |
| **overall** | 2,149 | 48.8% | 56.8% | 74.6% | 75.5% | 78.1% |

Four things worth reading off this.

**The easy band is worthless.** Where more than half the moves win, everything
scores 93-96% *including uniform random play*. Those positions are 45% of the
set and separate nothing. They are why the generator's quota now bands by
share: without one the suite fills with them.

**The sharp band is the whole measurement.** Where fewer than one move in ten
wins, random gets 1.3% and the nets get 51-60%. That is a fortyfold gap on
exactly the positions where a plan is needed rather than a shrug.

**Greedy is worse than random in one band and far better in another.** At
10-25% it scores 20.2% against random's 7.0%, but at 25-50% it scores 34.5%
against random's 34.9%. A one-ply static evaluation has a systematic blind
spot, not a uniform weakness, which is what makes it a poor anchor.

**Depth is not difficulty.** Every bot is *best* at two plies and shows no
trend after: v4-12h runs 88.6, 76.2, 77.6, 76.6, 75.6, 72.0, 76.9 from two
plies to eight. An eight-ply forced win with many winning moves is easier than
a two-ply one with a single answer. Sharpness is the hardness axis; ply count
is close to noise.

Hidden information barely moves any of them (v5-2h: 73.4% known, 79.5%
hidden). The positions are late and the ranges are small by then.

The ordering — v4-2h `74.6`, v5-2h `75.5`, v4-12h `78.1` — is worth less than
it looks. All three are within a few points, and this measures endgame
conversion only. A bot can convert every won endgame and still play a poor
opening. Read it beside the ladder, not instead of it.

## A benchmark is only as neutral as the games it came from

The same six bots, marked the same way, on two sets of proven endgame
positions. One set came from random play, the other from `v5-2h` playing
itself. Overall, and on the sharp band where fewer than one legal move in ten
keeps the win:

| | random-play set | v5 self-play set |
|---|---:|---:|
| v4-2h overall | `74.6%` | `65.6%` |
| v5-2h overall | `75.5%` | `57.0%` |
| v4-2h, under 10% | `51.3%` | `58.1%` |
| v5-2h, under 10% | `59.6%` | `48.9%` |

On random-play positions v5-2h beats v4-2h. On v5's own positions v4-2h beats
it by ten points, and v5 falls to last of the nets -- below `v3-trav`, three
architectures older. Same binaries, same 90-test build, same proof.

The cause is a selection effect in how positions are harvested, and it is
worth stating carefully because it is easy to get backwards. A position is
kept when the side to move has a proven forced win. If that player *converts*
the win, the game ends and the line yields one or two positions. If it
*misses*, the position stays won ply after ply and the line yields a dozen --
each a different position, so deduplication does not catch them, and every one
of them drawn from a line that player is misplaying. Missed wins are amplified;
converted wins are not. A self-play set is therefore concentrated on its own
author's blind spots.

That makes it an excellent adversarial probe of the bot that generated it, and
a poor yardstick for anyone else. The fix is one question per game, which
removes the amplification at the source; the standing advice is to generate
comparison sets from random play, or from a bot that is not under test.

Two things this does *not* explain, and they are worth keeping separate. The
first is that `v5-2h` tops the ladder at `1313` Elo against `v4-2h`'s `1060`
while converting fewer proven endgames on every set -- winning games and
finishing won positions are different skills, and a yardstick that collapsed
them would be hiding something rather than showing it. The second is that
`v4-12h` leads everywhere, on both sets and in every band, which is the least
surprising result here: it is trained six times longer than anything else in
the field.
