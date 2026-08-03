# warchest-engine

A verified rules engine for War Chest (2-player ranked configuration), in Rust,
with optional pyo3 bindings. This is the clean core extracted from an earlier
project — engine only, no training code.

## Layout

```
engine/          Rust crate, lib name `warchest`
  RULES.md       rules spec (source of truth)
  src/           actions, board, rng, rules, state, units, py (bindings)
  tests/         36 scenario tests + random-playout invariants
  examples/      coords.rs (hex coordinate dump)
  src/bin/       bench.rs (applies/sec, playouts/sec)
docs/
  ENGINE_FIXES.md  rule corrections found by replaying 1,112 real games
papers/          War Chest rulebook, ReBeL, TurboReBeL
```

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
