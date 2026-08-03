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
  examples/      coords.rs (hex coordinate dump)
  src/bin/       bench.rs (applies/sec, playouts/sec)
train/           train.py (PyTorch training loop)
docs/
  ENGINE_FIXES.md  rule corrections found by replaying 1,112 real games
  REBEL.md         the ReBeL agent: PBS design, CFR solver, deviations
papers/          War Chest rulebook, ReBeL, TurboReBeL
```

## Training

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install torch numpy maturin
cd engine && maturin develop --release && cd ..
.venv/bin/python train/train.py --minutes 30 --out runs/mine
```

War Chest turns out to be an unusually good fit for ReBeL. A player's private
state is exactly `(hand, face-down discards)` — the bag is derived from a public
reserve — and the reachable set has median 8 and p99 385 members, so CFR
enumerates information states exactly instead of approximating them with
particles. See `docs/REBEL.md`.

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
cargo test                          # 39 tests (the invariant playout takes ~100s)
cargo run --release --bin bench     # throughput, ~2.8M applies/sec/core
maturin develop --release           # python module `warchest` (Game)
```

The `python` feature is off by default; the pure-Rust API needs no pyo3.
