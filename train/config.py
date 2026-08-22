"""Production training knobs."""

import dataclasses
import os
import subprocess
from dataclasses import dataclass


@dataclass
class Cfg:
    minutes: float = 30.0
    snapshot_every: float = 15.0
    init_weights: str = ""

    batch: int = 1024
    lr: float = 1e-3
    lr_decay_frac: str = "0.33,0.67"
    # Gradient updates per generated row — Student of Games' "max grad updates
    # per example", which it runs at 10 for poker and 5 for Scotland Yard. A
    # solve yields one row now, so this is also updates per solve.
    replay_ratio: float = 8.0
    target_every: float = 5.0
    recent_mix: float = 0.5
    recent_frac: float = 0.2
    # Rows, and one row per solve. Student of Games holds 1M for both of its
    # imperfect-information games.
    cap: int = 1_000_000
    cfgs_per_row: int = 48
    # Student of Games' SoG(s, c): `s` expansions in all, `c` of them after
    # each regret update, so the solve runs ceil(s / c) updates. They are
    # distinct expansions: a phase draws trajectories until it has leaves the
    # round has not taken, so `s` sets the tree size and `batch` does not.
    s: int = 512
    c: float = 8.0
    # Regret updates one round of a solve carries. The tree is frozen for the
    # whole round and grows once at its end, from every leaf the round took, so
    # the per-round cost of re-describing an unchanged tree is paid once for the
    # whole round instead of once each -- and so is the leaf pass, which asks
    # the value network about every leaf at the head of the round and re-scales
    # what it said after. Eight is a sixth of the join work of one, on the same
    # solve. (`batch` above is the optimizer's, which is a different thing
    # entirely.)
    round_batch: int = 8
    # Round boundaries tree growth may pass through, 0 being today's limit.
    # Two converged solves disagree on the root value by 0.077 across it.
    rounds: int = 0
    # Student of Games' p_td1: the chance that a self-play row's value target is
    # the game's realised outcome instead of that solve's own CFR values. Table
    # S2 runs 0 for chess, poker and Scotland Yard and 0.2 for Go; 0.2 is what a
    # game this long needs, because a tree that never reaches a terminal has no
    # other source of ground truth.
    p_td1: float = 0.2
    # Student of Games' q_search and q_recursive: leaves drawn from each solve
    # and queued to be re-solved as roots of their own. This is the only way a
    # target is ever taken off the line of play.
    query_rate: float = 0.9
    recursive_rate: float = 0.1
    # What Student of Games runs: regret-matching+ with linearly-weighted
    # policy averaging, against simultaneous updates. `Solver::step` supplies
    # the simultaneous half.
    cfr: str = "sog"
    # Student of Games weights the two heads, `wv * huber + wp * cross_entropy`.
    # The value head is what CFR consumes, so it keeps weight one and the
    # policy -- which only steers the expansion phase -- comes in under it.
    # The paper's own numbers are 0.01 for poker and 0.05 for Scotland Yard,
    # and the cross entropy here runs 60--1000x the value Huber, so 0.05 is
    # what leaves the shared trunk answering to the value head.
    policy_w: float = 0.05
    # ReBeL's and Student of Games' off-policy exploration rate; both run 0.1.
    explore: float = 0.1
    # The horizon payoff per marker of lead. It is what carries the cold start:
    # a game cut at the play cap scores the win condition graded, which is a
    # real signal where a flat draw is none. `anneal_frac` takes it to zero,
    # and evaluation always runs the real game.
    cap_value: float = 0.15
    anneal_frac: float = 0.4
    random_draft: bool = True
    ladder_games: int = 40

    device: str = "cuda:1"
    # Cards the solve farm evaluates on. A round is split across them by call,
    # so each builds and runs a self-contained batch.
    gen_devices: str = "0,1"
    gen_solves: int = 8
    # Host threads that advance solves. Zero uses every physical CPU core. How
    # many solves are in flight is not a knob: each card admits them as its own
    # memory allows.
    gen_workers: int = 0
    train_stream_priority: int = -1

    out: str = ""
    seed: int = 1
    git: str = ""
    note: str = ""


BASELINE = Cfg()
CAST = {"int": int, "float": float,
        "bool": lambda v: v not in ("0", "false", "False", "")}


def parse(kvs):
    over = dict(kv.split("=", 1) for kv in kvs)
    kinds = {f.name: getattr(f.type, "__name__", f.type)
             for f in dataclasses.fields(Cfg)}
    bad = set(over) - set(kinds)
    if bad:
        raise SystemExit(f"no such knob: {sorted(bad)}")
    return {k: CAST.get(kinds[k], str)(v) for k, v in over.items()}


def knobs(cfg):
    """Every knob in Cfg order. `changed` is versus golden8 defaults."""
    d = cfg if isinstance(cfg, dict) else dataclasses.asdict(cfg)
    base = dataclasses.asdict(BASELINE)
    for f in dataclasses.fields(Cfg):
        v = d.get(f.name, base[f.name])
        yield f.name, v, f.name in base and v != base[f.name]


def git_sha():
    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", ".gitsha")
    try:
        with open(marker) as f:
            return f.read().strip() or "unknown"
    except OSError:
        pass
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--", ".", ":!runs"],
            capture_output=True)
        return sha + ("+dirty" if dirty.returncode != 0 else "")
    except Exception:
        return "unknown"
