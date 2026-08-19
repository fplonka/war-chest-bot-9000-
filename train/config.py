"""Production training knobs."""

import dataclasses
import os
import subprocess
from dataclasses import dataclass


@dataclass
class Cfg:
    minutes: float = 30.0
    warm_minutes: float = 5.0
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
    # The growing tree's budget. A round is six coin plays, so a tree has to
    # reach depth six to see the round boundary where beliefs reset — and at
    # about twenty actions a decision, only selective growth gets there. The
    # previous 256 nodes stopped growth after three of sixteen iterations,
    # which spent a growing-tree solver as a depth-one one.
    nodes: int = 8192
    expand: int = 8
    iters: int = 64
    # Far enough above `nodes` that a healthy tree never reaches it. A solve
    # that does is thrown away, so the number only buys how much work is spent
    # discovering that.
    node_cap: int = 65536
    config_cap: int = 256
    # Student of Games' q_search and q_recursive: leaves drawn from each solve
    # and queued to be re-solved as roots of their own. This is the only way a
    # target is ever taken off the line of play.
    query_rate: float = 0.9
    recursive_rate: float = 0.1
    # What Student of Games runs: regret-matching+ with linearly-weighted
    # policy averaging, against simultaneous updates. `Solver::step` supplies
    # the simultaneous half.
    cfr: str = "sog"
    # ReBeL's and Student of Games' off-policy exploration rate; both run 0.1.
    explore: float = 0.1
    temp: float = 2.0
    eval_mix: float = 1.0
    cap_value: float = 0.04
    anneal_frac: float = 0.4
    random_draft: bool = True
    warm_games: int = 96
    ladder_games: int = 40

    device: str = "cuda:1"
    # Cards the solve farm evaluates on. A round is split across them by call,
    # so each builds and runs a self-contained batch.
    gen_devices: str = "0,1"
    gen_solves: int = 8
    # Zero uses every physical CPU core.
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
