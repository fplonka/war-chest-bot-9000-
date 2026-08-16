"""Production training knobs."""

import dataclasses
import os
import subprocess
from dataclasses import dataclass


@dataclass
class Cfg:
    minutes: float = 30.0
    warm_minutes: float = 5.0
    snapshot_every: float = 6.0
    init_weights: str = ""

    batch: int = 1024
    lr: float = 1e-3
    lr_decay_frac: str = "0.33,0.67"
    train_gen_ratio: float = 4.0
    recent_mix: float = 0.5
    recent_frac: float = 0.2
    cap: int = 2_000_000
    cfgs_per_row: int = 48
    depth: int = 1
    iters: int = 64
    cfr: str = "dcfr"
    explore: float = 0.25
    temp: float = 2.0
    eval_mix: float = 1.0
    cap_value: float = 0.04
    anneal_frac: float = 0.4
    random_draft: bool = True
    warm_games: int = 96
    ladder_games: int = 40

    device: str = "cuda:1"
    gpu_devices: str = "0,1"
    gpu_workers: int = 36
    gpu_inflight: int = 32
    gpu_chunk: int = 1024
    gpu_drain_seconds: float = 20.0
    gpu_publish_steps: int = 16
    train_stream_priority: int = -1

    out: str = ""
    note: str = ""
    seed: int = 1
    git: str = ""


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
