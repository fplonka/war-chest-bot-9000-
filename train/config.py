"""Knobs of a training run. Defaults are gpu_golden8."""

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
    random_draft: bool = True
    warm_games: int = 96
    ladder_games: int = 100

    device: str = "cuda:1"
    gpu_devices: str = "0,1"
    gpu_workers: int = 36
    gpu_actors: int = 128
    gpu_inflight: int = 32
    gpu_chunk: int = 1024
    gpu_drain_seconds: float = 20.0
    gpu_publish_steps: int = 16
    train_stream_priority: int = -1

    out: str = ""
    seed: int = 1
    matmul_precision: str = ""
    git: str = ""


BASELINE = Cfg()
# Not part of the run's name.
SKIP = {"out", "git", "matmul_precision"}
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


def label(over):
    parts = [str(v) if isinstance(v, str) else f"{k}{v}"
             for k, v in sorted(over.items()) if k not in SKIP]
    return "-".join(parts) or "base"


def delta(cfg):
    base = dataclasses.asdict(BASELINE)
    d = cfg if isinstance(cfg, dict) else dataclasses.asdict(cfg)
    return {k: v for k, v in d.items()
            if k not in SKIP and k in base and base[k] != v}


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
