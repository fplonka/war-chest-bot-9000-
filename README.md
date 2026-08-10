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
webui/           play.py + index.html: browser UI for playing a trained agent
play.sh          one-liner: build the extension, serve the UI, open the browser
train/           config.py   every knob of a run, one object; the experiments we run
  exp.py         run an experiment end to end: arms x seeds -> ladder -> report
  train.py       PyTorch training loop, snapshots on a timer, judges nothing
  value_net.py   the value network itself, shared by every tool that loads one
  truth.py       a frozen set of solved positions, and any checkpoint's error on it
  ladder.py      round robin over runs' snapshots, plus Greedy -> Elo
  report.py      one self-contained HTML page per run or comparison
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
.venv/bin/python train/exp.py run dcfr --seeds 2     # arms, seeds, ladder, report
.venv/bin/python train/exp.py ls                     # every run, newest first
```

An experiment is declared in `train/config.py` as a list of arms, and an arm is
only its *delta* from the baseline. `exp.py` runs each arm at each seed, rates
every resulting checkpoint in one ladder, and writes a self-contained HTML page
— no manual step anywhere in that chain. One arm at a time is
`train/train.py --config <json>` or `--set knob=value`.

Not everything needs a training run. `train/offline.py` compares architectures
on a frozen replay dump, and `engine/examples/solvererr.rs` answers the
regret-rule and iteration-count questions without training anything at all. Use
those first; a ladder is the confirmation, not the search.

`train/truth.py` scores a checkpoint against a frozen set of positions solved to
convergence — one number, seconds, no variance — but read its module docstring
before believing it. The targets are the fixed point of the *network that built
the set*, and measurement shows the set ranks that network's neighbourhood above
genuinely stronger ones. It tracks one run's progress; it does not compare arms.
Only the ladder does that.

## Playing against a trained agent

```bash
./play.sh                      # newest final checkpoint, opens the browser
./play.sh --ckpt runs/t64_h384_dg64_s11/snap_05.pt
```

The default is the newest `runs/*/ckpt_final.pt`, falling back to the newest
`runs/*/snap_*.pt` final snapshot (the long runs save `snap_XX.pt`); pass
`--ckpt` to pick a specific checkpoint. `--depth`/`--iters` default to the
training configuration (2/16).

You play white with the rulebook's starter army (Swordsman, Pikeman,
Crossbowman, Light Cavalry) against the fixed black army (Archer, Cavalry,
Lancer, Scout). Round-start draws and the agent's moves resolve automatically;
every decision that is yours — including triggered follow-ups like the
Swordsman's free move — appears as a clickable legal action. The agent solves
the depth-limited subgame at each of its decisions with the same CFR-average
configuration evaluation uses, so it plays like the checkpoint's `eval.json`
says it plays. The agent's hand, bag and face-down discards are hidden from
the browser — including the coin it spends face-down on Pass / Claim
initiative / Recruit, which the game log never reveals — and only public
counts are shown.

The session object behind the UI is `warchest.LiveGame` (`engine/src/live.rs`),
which mirrors the self-play loop: public beliefs over both players' configs,
chance resolution from the true bag, and a Bayes update on the *public*
observation after every action. The one deliberate divergence from self-play is
that the human's belief update assumes a uniform behaviour model for the human
— the agent has no model of how a person plays, and a model that assumed
agent-like play could drop the true config from the belief support.

A run saves the network every few minutes and judges nothing while it trains:
it produces checkpoints and stops. `train/ladder.py` rates them afterwards, so
a measurement can be rerun at any sample size without regenerating anything.
What a run reports is strength against minutes trained, with Greedy at 0.

Sample size is the thing to get right. A pairing of 100 games resolves nothing
finer than about 70 Elo, 1,000 games about 22, and 5,000 about 10 — while a game
costs on the order of a hundred solves against the million a training run
generates. So the ladder puts most of its games on the pairing the experiment is
about (`--focus`, `--focus-games`) and enough everywhere else to place a player.
Ladders here used to run 30 to 100 games; the architecture comparisons decided
that way were inside the noise.

War Chest turns out to be an unusually good fit for ReBeL. A player's private
state is `(hand, face-down discards, pending forced-play coin)` — the bag is
derived from a public reserve — and the reachable set has median 22 and p99 567
members with the full draft pool, so CFR enumerates information states exactly
instead of approximating them with particles. The value network is a function of that exact private state, not of a
summary of it: `docs/REBEL.md` §4 explains why the alternative is not an
approximation but a different game.

Thirty minutes on an 8-core M1 takes the agent from 356 Elo to 852, against 174
for the handcrafted Greedy reference and 0 for random play — and shows it
gaining almost nothing after the first seventeen (`runs/elo01`). See
`docs/PERF.md` for how the generation loop got fast enough for that to fit in
half an hour.

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
The v5 GPU executor's deterministic throughput gate is
`examples/wave_tape.rs`; it uses production roots and reports the complete work
distribution before timing. `docs/GPU_PERF_GOAL.md` defines the real training
target and corrected baseline; `docs/GPU_ARCHITECTURE.md` describes the active
replacement architecture and verification runbook.

Build it with `--features prof` for a per-phase breakdown. Its `games depth
iters threads` arguments default to the trainer's settings, so a throughput
number is only comparable to another taken at the same ones — `iters` in
particular is most of the cost.

The `python` feature is off by default; the pure-Rust API needs no pyo3.
