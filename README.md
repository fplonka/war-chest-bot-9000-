# warchest-engine

A verified rules engine for War Chest (2-player ranked configuration), in Rust,
with pyo3 bindings — plus a ReBeL agent trained on top of it.

## Layout

```
engine/          Rust crate, lib name `warchest`
  RULES.md       rules spec (source of truth)
  src/           actions, board, rng, rules, state, units, py (bindings)
  src/           rebel, search, selfplay, net  (the ReBeL agent)
  src/           arena, bot, policy  (the referee, a bot, and a node's policy)
  tests/         36 scenario tests, playout invariants, PBS correctness tests
  examples/      coords.rs (hex dump), featstats.rs (feature ranges),
                 cfgvalue.rs (how far the value separates configs)
  src/bin/       bench.rs (applies/sec, playouts/sec), bot.rs (an arena bot)
webui/           play.py + index.html: browser UI for playing a trained agent
play.sh          one-liner: build the extension, serve the UI, open the browser
train/           train.py    PyTorch training loop, snapshots on a timer
  value_net.py   the value network itself, shared by every tool that loads one
  mirror.py      the board's 180-degree rotation, which is how the second
                 canonical seat view of a row is produced
  diagnose.py    model-free check on how learnable a dump's targets are
  dump.py        reading a dumped replay buffer
  gpu_batch.py   replay rows -> canonical query batch, expanded on the device
tools/           box.sh      the GPU box: sync, build, run, pull a run back
  arena.py       archive bots; ladder them, and score them on proven endgames
  monitor.py     live dashboard over runs/ and arena/, served from disk
  farmbench.py   solves/s on a fixed corpus, for ranking builds
  genrate.sh     solves/s in a real short run, which is the number that counts
  probe.sh       the card and host profiles behind docs/REDESIGN.md
  netablate.py   what each part of the value network costs, and buys
docs/
  ENGINE_FIXES.md  rule corrections found by replaying 1,112 real games
  REBEL.md         the ReBeL agent: PBS design, CFR solver, the value network
  REDESIGN.md      what a solve costs, what binds, and what to rip out
  ARENA.md         bots, the referee protocol, and how a ladder is run
```

Not in the repo: `papers/` (gitignored) is where the AEG rulebook and the
reference papers are kept locally — ReBeL (arXiv:2007.13544), Student of Games
(arXiv:2112.03178) and DeepStack (arXiv:1701.01724) are the three the value
target convention comes from. The rulebook is a commercial product, so it is
not redistributed here.

## Training

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install torch numpy maturin
cd engine && maturin develop --release && cd ..
.venv/bin/python train/train.py out=mine minutes=30
python3 tools/monitor.py       # watch it live at http://127.0.0.1:8420
```

## Playing against a trained agent

```bash
./play.sh                      # the newest bot under bots/, opens the browser
./play.sh bots/v5-2h
```

You play a **bot** — the same directory the arena ladders — and it searches the
way its `bot.json` says it does, so the agent in the browser is the agent the
ladder rated. Only bots this engine revision can load will play; an older
architecture carries its own binary and is played through the arena.

You play white with the rulebook's starter army (Swordsman, Pikeman,
Crossbowman, Light Cavalry) against the fixed black army (Archer, Cavalry,
Lancer, Scout). Round-start draws and the agent's moves resolve automatically;
every decision that is yours — including triggered follow-ups like the
Swordsman's free move — appears as a clickable legal action. At each decision
the agent grows a GT-CFR tree along its sampled strategy and plays from the
final CFR average, with the same node and iteration budgets the arena uses.
The agent's hand, bag and face-down discards are hidden from
the browser — including the coin it spends face-down on Pass / Claim
initiative / Recruit, which the game log never reveals — and only public
counts are shown.

Behind the UI is the arena's own referee (`engine/src/arena.rs`): `play.py`
holds the true game and the dice, the bot is the same subprocess a ladder runs,
and you are the other seat. There is no second implementation of a live game —
the agent you play is the agent the ladder rated, down to the binary.

A run saves the network every few minutes and judges nothing while it trains.
When it ends, every snapshot is archived as a **bot** — a directory holding a
frozen binary, its weights and a manifest — and the arena plays them against
each other and against the handcrafted Greedy reference. What a run reports is
strength against minutes trained. `train.py` does this automatically:

```bash
python tools/arena.py pack runs/mine            # bots/mine.init, bots/mine.s1, ...
python tools/arena.py ladder bots/mine.* bots/greedy --games 200
```

The same command rates one architecture against another, because a bot built at
an old revision keeps playing after the source that produced it has been
rewritten. See `docs/ARENA.md`.

War Chest turns out to be an unusually good fit for ReBeL. A player's private
state is `(hand, face-down discards, pending forced-play coin)` — the bag is
derived from a public reserve — and the reachable set has median 22 and p99 567
members with the full draft pool, so CFR enumerates information states exactly
instead of approximating them with particles. The value network is a function of
that exact private state, not of a summary of it: `docs/REBEL.md` §5 explains
why the alternative is not an approximation but a different game.

Thirty minutes on an 8-core M1 took the v4 agent from 356 Elo to 852, against 174
for the handcrafted Greedy reference and 0 for random play — and shows it
gaining almost nothing after the first seventeen (`runs/elo01`).

## Design

The game is a sequence of **decision nodes**. `State::legal_actions` returns the
actions for whoever is to act (including chance draws) and `State::apply`
returns the successor state. All randomness enters through chance-node
`DrawCoin` actions; the core is RNG-free and deterministic, so a replay can
force observed draws.

## Provenance

The rules were verified against warchestonline.com by replaying real games
action-by-action: **1,112/1,112 in-scope games (93,566 actions) clean**, plus a
holdout of **347/347 unseen games (27,091 actions) clean**. Every rule fix found
that way has a scenario test — see `docs/ENGINE_FIXES.md`.

## Build

```bash
cd engine
cargo test                          # solver and engine tests
cargo run --release --bin bench     # engine throughput, ~2.8M applies/sec/core
maturin develop --release           # python module `warchest` (Game)
```

Generation throughput is `tools/farmbench.py` against a fixed corpus, and
`tools/genrate.sh` in a real short run. Throughput is only comparable at
identical search budgets.

The `python` feature is off by default; the pure-Rust API needs no pyo3.
