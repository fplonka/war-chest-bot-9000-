"""Every knob of a training run, as one object, plus the experiments we run.

Runs used to be shell lines with sixty flags, and an experiment was whatever
that line said on the day. Nothing could diff two runs, so every NOTES.md
hand-copied its own settings and an arm that accidentally differed in two
places was undetectable afterwards.

So: one `Cfg`, one `BASELINE`, and an experiment names *only what it changes*.
`delta()` recovers those changes from a finished run's `log.json`, which is what
the report prints and what makes a claim about an arm checkable rather than
remembered.

    from config import BASELINE, arms
    cfg = replace(BASELINE, iters=128)

Adding a knob means adding a field here. Defaults are the production
configuration: a bare `BASELINE` is the run every arm is measured against.
"""

import dataclasses
import json
import subprocess
from dataclasses import dataclass, replace


@dataclass
class Cfg:
    # ------------------------------------------------------------- schedule
    minutes: float = 60.0
    warm_minutes: float = 5.0      # absolute; the warm phase's data is discarded
    warm_frac: float = 0.2         # used only when warm_minutes < 0
    snapshot_every: float = 9.0    # minutes; snapshots cost a file write
    init: str = ""                 # optional checkpoint to start from

    # -------------------------------------------------------------- network
    hidden: int = 384
    head: int = 0                  # readout width; 0 means "same as hidden"
    dg: int = 64                   # config embedding = rank of the private-state dependence
    rank: int = 64                 # rank of the value readout's inner product
    de: int = 32                   # card embedding width
    nres: int = 1                  # holding residual blocks
    pub: str = ""                  # tower widths, comma lists; empty = classic shape
    hmlp: str = ""
    card: str = ""
    slot: str = ""

    # ----------------------------------------------------------------- loss
    # Both heads train the shared trunk, so both change the value network and
    # both have to be gated as their own change. Off until measured.
    policy: float = 0.0            # weight on the policy head (labels are free)
    aux: float = 0.0               # weight on the auxiliary heads

    # ------------------------------------------------------------ optimiser
    batch: int = 1024
    lr: float = 1e-3
    lr_decay_frac: str = "0.33,0.67"   # halve the lr at these fractions of the ReBeL phase
    train_gen_ratio: float = 4.0
    recent_mix: float = 0.5        # fraction of a batch drawn from the newest slice
    recent_frac: float = 0.2       # how big that slice is
    cap: int = 2_000_000           # replay capacity: an algorithmic knob, not a memory one
    cfgs_per_row: int = 48
    no_augment: bool = False

    # --------------------------------------------------------------- search
    # depth 1 puts zero opponent decision nodes in the subgame, which reduces
    # CFR to 1-ply value iteration over the network. 2 is the minimum that is
    # actually ReBeL.
    depth: int = 2
    # CFR iterations per subgame. Against a converged T=512 reference
    # (`examples/solvererr.rs`) the mean |error| in the root value is 0.0098 at
    # T=8, 0.0036 at T=16, 0.0016 at T=32. That is *bias*, not noise -- the same
    # position gives the same wrong number every time -- so the network fits it
    # happily and converges to the fixed point of the under-solved operator. No
    # loss curve can show that.
    iters: int = 64
    cfr: str = "linear"            # linear, plus, dcfr, pcfr, sapcfr (docs/REBEL.md)
    warm: float = 0.0              # iterations the policy head's seed is worth
    explore: float = 0.25
    temp: float = 2.0
    eval_mix: float = 0.5
    mc_mix: float = 0.0            # blend of realised outcome into the target; 0 = plain ReBeL
    # Horizon payoff per marker of differential, annealed to zero over
    # `anneal_frac` of the ReBeL phase so the shipped checkpoint is fitted to
    # the real game. Each side has 6 markers, so this must stay far below a
    # real win or stalling the clock becomes a competing win condition.
    cap_value: float = 0.04
    anneal_frac: float = 0.4
    random_draft: bool = False
    warm_games: int = 96
    rebel_games: int = 48

    # ------------------------------------------------------------- hardware
    device: str = "cuda"
    gpu: bool = True
    gpu_devices: str = "0,1"
    gpu_workers: int = 36
    gpu_actors: int = 128
    gpu_inflight: int = 32
    gpu_chunk: int = 1024
    gpu_drain_seconds: float = 20.0
    gpu_publish_steps: int = 16
    train_stream_priority: int = -1

    # ----------------------------------------------------------------- runs
    out: str = "runs/latest"
    seed: int = 1
    dump_buffer: str = ""          # every gate run should set this; see truth.py
    # Kill a run whose generation has collapsed or whose value function has gone
    # flat, rather than spending the hour finding out. See train.py::check_alive.
    abort_below_sps: float = 50.0
    abort_below_spread: float = 0.02
    matmul_precision: str = ""     # recorded, not set
    git: str = ""                  # recorded, not set


BASELINE = Cfg()


# Each experiment is a list of arms, and an arm is only its delta from
# BASELINE. The empty delta is the control and every experiment needs one --
# comparing an arm against a *previous* run's baseline compares two runs that
# differ in the code as well as the config.
EXPERIMENTS = {
    "dcfr":     [{}, {"cfr": "dcfr"}],
    "aux":      [{}, {"aux": 0.3}],
    "policy":   [{}, {"policy": 0.3}],
    "iters":    [{}, {"iters": 32}, {"iters": 128}],
    # Does a short run rank changes the way a long one does? If it does, every
    # experiment above costs a quarter of what it costs today. Run this first.
    "cadence":  [{"minutes": 15, "snapshot_every": 5},
                 {"minutes": 15, "snapshot_every": 5, "cfr": "dcfr"},
                 {"minutes": 15, "snapshot_every": 5, "iters": 32}],
}


def arms(name):
    """An experiment's arms as `(label, Cfg)`, its control first.

    An arm is named for what it changes *within its experiment*, not for its
    whole delta from BASELINE: when every arm shortens the run to 15 minutes,
    that is the experiment's setting and only the rest tells the arms apart.
    """
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
    """An arm's name: what it changes, or `base` when it changes nothing.

    A string value names itself (`cfr="dcfr"` is the `dcfr` arm); a number needs
    its knob to mean anything (`iters=32` is the `iters32` arm).
    """
    parts = [str(v) if isinstance(v, str) else f"{k}{v}" for k, v in sorted(d.items())]
    return "-".join(parts) or "base"


def delta(cfg):
    """What a config changes from BASELINE, ignoring per-run bookkeeping."""
    skip = {"out", "seed", "git", "matmul_precision", "dump_buffer"}
    base = dataclasses.asdict(BASELINE)
    d = cfg if isinstance(cfg, dict) else dataclasses.asdict(cfg)
    return {k: v for k, v in d.items()
            if k not in skip and k in base and base[k] != v}


def git_sha():
    """The commit a run was launched from, `+dirty` when the tree is not clean.

    A number whose code you cannot recover is a number you cannot check, so
    every run records this and `exp.py` refuses to gate from a dirty tree.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def load(path):
    """A `Cfg` from a run's `log.json`, or from a config file exp.py wrote."""
    with open(path) as f:
        d = json.load(f)
    d = d.get("cfg", d)
    known = {f.name for f in dataclasses.fields(Cfg)}
    return Cfg(**{k: v for k, v in d.items() if k in known})
