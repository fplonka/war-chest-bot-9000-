"""Every knob of a training run, as one object, plus the experiments we run.

One `Cfg`, one `BASELINE`, and an experiment names only what it changes.
`delta()` recovers those changes from a finished run's `log.json`.
"""

import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass, replace


@dataclass
class Cfg:
    minutes: float = 30.0
    warm_minutes: float = 5.0
    snapshots: int = 3
    init_weights: str = ""

    hidden: int = 384
    head: int = 0
    dg: int = 64
    rank: int = 64
    de: int = 32
    nres: int = 1
    pub: str = ""
    hmlp: str = ""
    card: str = ""
    slot: str = ""

    policy: float = 0.0
    aux: float = 0.0

    batch: int = 1024
    lr: float = 1e-3
    lr_decay_frac: str = "0.33,0.67"
    train_gen_ratio: float = 4.0
    recent_mix: float = 0.5
    recent_frac: float = 0.2
    cap: int = 2_000_000
    cfgs_per_row: int = 48
    no_augment: bool = False

    depth: int = 2
    iters: int = 64
    cfr: str = "linear"
    warm: float = 0.0
    explore: float = 0.25
    temp: float = 2.0
    eval_mix: float = 0.5
    cap_value: float = 0.04
    anneal_frac: float = 0.4
    random_draft: bool = False
    warm_games: int = 96

    device: str = "cuda:1"
    gpu_devices: str = "0,1"
    gpu_workers: int = 36
    gpu_actors: int = 128
    gpu_inflight: int = 32
    gpu_chunk: int = 1024
    gpu_drain_seconds: float = 20.0
    gpu_publish_steps: int = 16
    train_stream_priority: int = 0

    out: str = "runs/latest"
    seed: int = 1
    dump_buffer: str = ""
    experiment: str = ""
    arm: str = ""
    matmul_precision: str = ""
    git: str = ""


BASELINE = Cfg()

EXPERIMENTS = {
    "smoke":   [{"minutes": 6, "warm_minutes": 2}],
    "sanity":  [{}],
    "seat":    [{}],
    "wp":      [{}],
    "explore": [{}],
    "adam":    [{}],
    "zsum":    [{}],
}


def arms(name):
    """An experiment's arms as `(label, Cfg)`, its control first."""
    if name not in EXPERIMENTS:
        raise SystemExit(f"unknown experiment {name!r}; have {sorted(EXPERIMENTS)}")
    known = {f.name for f in dataclasses.fields(Cfg)}
    out = []
    control = EXPERIMENTS[name][0]
    for d in EXPERIMENTS[name]:
        bad = set(d) - known
        if bad:
            raise SystemExit(f"{name}: no such knob {sorted(bad)}")
        own = {k: v for k, v in d.items() if control.get(k) != v}
        out.append((label(own), replace(BASELINE, **d)))
    return out


def label(d):
    parts = [str(v) if isinstance(v, str) else f"{k}{v}" for k, v in sorted(d.items())]
    return "-".join(parts) or "base"


def delta(cfg):
    skip = {"out", "seed", "git", "matmul_precision", "dump_buffer",
            "experiment", "arm"}
    base = dataclasses.asdict(BASELINE)
    d = cfg if isinstance(cfg, dict) else dataclasses.asdict(cfg)
    return {k: v for k, v in d.items()
            if k not in skip and k in base and base[k] != v}


def git_sha():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--", ".", ":!runs"],
            capture_output=True)
        return sha + ("+dirty" if dirty.returncode != 0 else "")
    except Exception:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", ".gitsha")) as f:
                return f.read().strip() or "unknown"
        except OSError:
            return "unknown"


def load(path):
    with open(path) as f:
        d = json.load(f)
    d = d.get("cfg", d)
    known = {f.name for f in dataclasses.fields(Cfg)}
    unknown = set(d) - known
    if unknown:
        raise SystemExit(f"{path}: unknown config fields {sorted(unknown)}")
    return Cfg(**d)
