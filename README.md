# warchest-engine

A verified rules engine for War Chest (2-player ranked configuration), in Rust,
with pyo3 bindings — plus a ReBeL agent trained on top of it.

## Layout

```
engine/          Rust crate, lib name `warchest`
  RULES.md       rules spec (source of truth)
  src/           actions, board, rng, rules, state, units, py (bindings)
  src/           rebel, search, selfplay, net  (the ReBeL agent)
  tests/         36 scenario tests, playout invariants, PBS correctness tests
  examples/      coords.rs (hex dump), featstats.rs (feature ranges),
                 solvererr.rs (CFR target error vs iteration count),
                 cfgvalue.rs (how far the value separates configs)
  src/bin/       bench.rs (applies/sec, playouts/sec)
train/           train.py    PyTorch training loop, snapshots on a timer
  value_net.py   the value network itself, shared by every tool that loads one
  ladder.py      round robin over a run's snapshots, plus Greedy and Random -> Elo
  plot.py        the four panels a run is read from
  offline.py     fit architectures to a frozen replay dump (noise-free A/B)
  diagnose.py    model-free check on how learnable a dump's targets are
  dump.py        reading a dumped replay buffer
  mirror.py      the board's 180-degree symmetry, as a data augmentation
docs/
  ENGINE_FIXES.md  rule corrections found by replaying 1,112 real games
  REBEL.md         the ReBeL agent: PBS design, CFR solver, deviations
  PERF.md          how the generation loop got ~10x faster, and what didn't work
```

Not in the repo: `papers/` (gitignored) is where the AEG rulebook, the ReBeL
paper (arXiv:2007.13544) and the TurboReBeL paper are kept locally. The rulebook
is a commercial product, so it is not redistributed here.

## Training

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install torch numpy maturin
cd engine && maturin develop --release && cd ..
.venv/bin/python train/train.py --minutes 30 --out runs/mine
.venv/bin/python train/plot.py runs/mine
```

A run saves the network every few minutes and judges nothing while it trains.
When it ends, `train/ladder.py` plays every snapshot against every other one,
against the handcrafted Greedy bot and against Random, and fits an Elo to each —
so what a run reports is strength against minutes trained, with Random at 0.
`train.py` runs it automatically; `python train/ladder.py runs/mine --games 200`
reruns it with more games.

War Chest turns out to be an unusually good fit for ReBeL. A player's private
state is exactly `(hand, face-down discards)` — the bag is derived from a public
reserve — and the reachable set has median 8 and p99 385 members, so CFR
enumerates information states exactly instead of approximating them with
particles. The value network is a function of that exact private state, not of a
summary of it: `docs/REBEL.md` §4 explains why the alternative is not an
approximation but a different game.

Ten minutes on an 8-core M1 is enough to beat the handcrafted Greedy reference
370-1-29. See `docs/PERF.md` for how the generation loop got there.

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
cargo test                          # 55 tests (the solver oracle takes ~85s)
cargo run --release --bin bench     # engine throughput, ~2.8M applies/sec/core
cargo run --release --bin rebelbench -- weights.bin   # generation throughput
maturin develop --release           # python module `warchest` (Game)
```

`rebelbench` runs the ReBeL generation loop without Python, on weights exported
by `train/export_weights.py`, and is what `docs/PERF.md`'s numbers come from.
Build it with `--features prof` for a per-phase breakdown. Its `games depth
iters threads` arguments default to the trainer's settings, so a throughput
number is only comparable to another taken at the same ones — `iters` in
particular is most of the cost.

The `python` feature is off by default; the pure-Rust API needs no pyo3.
