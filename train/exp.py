"""Run an experiment end to end, and list what has been run.

    python train/exp.py run dcfr --seeds 2      # both arms, both seeds, judged
    python train/exp.py ls                      # every run, newest first
    python train/exp.py judge runs/dcfr-*       # re-judge, more games

An experiment is declared in `config.py` as a list of arms, and an arm is only
its delta from `BASELINE`. This runs each arm at each seed, then rates every
resulting checkpoint in one ladder, then writes one HTML page. Nothing in that
chain is a manual step, because a manual step is one that gets skipped at two
in the morning and cannot be reconstructed afterwards.

Two seeds by default, and that is not caution. A ladder's error bars say how
well two *checkpoints* compare; they say nothing about whether a second run of
the same config would have produced that checkpoint. Run-to-run variation is
about ±0.05 in score on a short run -- larger than most effects worth chasing --
and the only instrument for it is running the config again. An arm that wins on
one seed and loses on the other has not won.

The ladder concentrates its games on the arms' final checkpoints, since that is
the comparison the experiment exists to make; everything else gets enough games
to be placed. See `ladder.py` for why that is allowed and what it buys.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import ladder
import report

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "runs"


def launch(cfg, out):
    """One training run, as its own process. Returns True if it finished."""
    os.makedirs(out, exist_ok=True)
    path = f"{out}/config.json"
    ladder.write_json(path, dataclasses.asdict(cfg))
    cmd = [sys.executable, f"{HERE}/train.py", "--config", path]
    print(f"\n=== {out} ===\n{' '.join(cmd)}", flush=True)
    t = time.time()
    # Its own process, so failures have an exact return code and an incomplete
    # arm cannot leak Python/CUDA state into the next experiment.
    with open(f"{out}/train.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"=== {out}: {'ok' if rc == 0 else f'FAILED rc={rc}'} "
          f"in {(time.time() - t) / 60:.1f} min ===", flush=True)
    if rc != 0:
        print(f"    tail -20 {out}/train.log", flush=True)
    return rc == 0


def comparisons_from_metadata(runs):
    """Explicit `(control, candidate)` pairs from experiment and seed metadata."""
    records = []
    for run in runs:
        with open(f"{run}/log.json") as f:
            cfg = json.load(f).get("cfg", {})
        records.append((run, cfg))
    pairs = []
    keys = sorted({(c.get("experiment", ""), c.get("seed")) for _, c in records})
    for experiment, seed in keys:
        group = [(r, c) for r, c in records
                 if (c.get("experiment", ""), c.get("seed")) == (experiment, seed)]
        controls = [r for r, c in group if c.get("is_control")]
        if len(controls) != 1:
            raise ValueError(
                f"experiment {experiment!r} seed {seed}: expected one control run, "
                f"got {controls}")
        pairs.extend((controls[0], r) for r, c in group if not c.get("is_control"))
    return pairs


def judge(runs, games, comparison_games, gpu, seed=7, labels=None, fixed_draft=False):
    """Rate every arm's snapshots in one ladder, then write the page.

    `labels` can restrict which snapshots enter; by default every checkpoint is
    evaluated and the sparse graph remains linear in their number.
    """
    comparisons = comparisons_from_metadata(runs)
    result = ladder.run(runs, out=runs[0], games=games, comparisons=comparisons,
                        comparison_games=comparison_games, gpu=gpu, seed=seed, labels=labels,
                        random_draft=not fixed_draft)
    for r in runs:
        # Each run gets its own page, and the comparison gets one more.
        if r != runs[0]:
            ladder.write_json(f"{r}/ladder.json", result)
        report.write([r], f"{r}/report.html")
    if len(runs) > 1:
        common = os.path.commonprefix([os.path.basename(r.rstrip("/")) for r in runs])
        out = f"{RUNS}/{common.rstrip('-') or 'compare'}.html"
        report.write(runs, out)


def cmd_run(args):
    sha = config.git_sha()
    if sha.endswith("+dirty") and not args.force:
        raise SystemExit(
            "the tree is dirty, so this run's numbers could not be reproduced from "
            "any commit. Commit first, or pass --force if you mean it.")
    arms = config.arms(args.name)
    # Keep a seed's control and candidates adjacent in wall-clock time. This
    # blocks slow host drift (temperature, shared-machine load) by the same seed
    # the direct comparison uses, instead of running every control first and
    # every candidate hours later.
    todo = [(f"{args.name}-{lab}" + (f"-s{s}" if args.seeds > 1 else ""),
             dataclasses.replace(cfg, seed=s, experiment=args.name, arm=lab,
                                 is_control=(i == 0)))
            for s in range(1, args.seeds + 1)
            for i, (lab, cfg) in enumerate(arms)]
    print(f"[exp] {args.name} at {sha}: {len(todo)} runs "
          f"({len(config.arms(args.name))} arms x {args.seeds} seeds)")
    for name, cfg in todo:
        print(f"  {name:<28s} {config.delta(cfg) or 'baseline'}")
    train_min = sum(c.minutes for _, c in todo)
    # Per run: a checkpoint chain and its two Greedy anchors. Each candidate
    # final also plays its same-seed control final.
    n = len(todo)
    per = (config.BASELINE.snapshots + 1 if args.labels == "all"
           else len(args.labels.split(",")))
    k = n * per + 1
    comparisons = (len(arms) - 1) * args.seeds
    anchors = n * min(per, 2)
    played = n * max(per - 1, 0) + anchors + comparisons
    ladder_games = ((played - comparisons) * args.games
                    + comparisons * args.comparison_games)
    print(f"[exp] {train_min:.0f} min of training, then a {k}-player ladder: "
          f"{played} pairings, {ladder_games:,} games "
          f"(~{ladder_games / 3 / 60:.0f} min)")
    if args.dry_run:
        return
    done = []
    for name, cfg in todo:
        out = f"{RUNS}/{name}"
        cfg = dataclasses.replace(cfg, out=out,
                                  dump_buffer=f"{out}/buf.npz" if args.dump else "")
        if not launch(cfg, out):
            raise SystemExit(f"[exp] {out} failed; experiment aborted")
        done.append(out)
    judge(done, args.games, args.comparison_games, gpu=config.BASELINE.gpu, seed=args.seed,
          labels=None if args.labels == "all" else args.labels.split(","),
          fixed_draft=args.fixed_draft)


def cmd_judge(args):
    judge(args.runs, args.games, args.comparison_games, gpu=args.gpu, seed=args.seed,
          labels=None if args.labels == "all" else args.labels.split(","),
          fixed_draft=args.fixed_draft)


def cmd_ls(args):
    """Every run that left a log, newest first: what it changed and how it did.

    The record is `log.json` itself -- there is no second database to keep in
    step with it, and a run that was never written down never happened.
    """
    rows = []
    for d in sorted(os.listdir(RUNS)):
        try:
            with open(f"{RUNS}/{d}/log.json") as f:
                log = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(log, dict):
            continue        # a log from before the run recorded its own config
        cfg, eps = log.get("cfg", {}), log.get("epochs", [])
        elo = ""
        try:
            with open(f"{RUNS}/{d}/ladder.json") as f:
                lad = json.load(f)
            fin = next((p for p in lad["players"] if p["name"].endswith(".final")), None)
            if fin:
                elo = f"{fin['elo']:.0f}±{fin['se']:.0f}"
        except (OSError, json.JSONDecodeError):
            pass
        rows.append((os.path.getmtime(f"{RUNS}/{d}/log.json"), d,
                     cfg.get("git", "?"), f"{eps[-1]['t'] / 60:.0f}m" if eps else "-",
                     elo, ", ".join(f"{k}={v}" for k, v in config.delta(cfg).items())))
    rows.sort(reverse=True)
    print(f"{'run':<34s} {'commit':<12s} {'len':>5s} {'final elo':>10s}  delta")
    for _, d, sha, mins, elo, delta in rows[:args.limit]:
        print(f"{d:<34s} {sha:<12s} {mins:>5s} {elo:>10s}  {delta or 'baseline'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run every arm of an experiment, then judge it")
    r.add_argument("name", help=f"one of {sorted(config.EXPERIMENTS)}")
    r.add_argument("--seeds", type=int, default=2)
    r.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    r.add_argument("--force", action="store_true", help="allow a dirty tree")
    r.add_argument("--dump", action="store_true",
                   help="write each run's replay buffer; a few GB per arm")
    r.set_defaults(fn=cmd_run)

    j = sub.add_parser("judge", help="rate finished runs (again, or at more games)")
    j.add_argument("runs", nargs="+")
    j.add_argument("--gpu", action="store_true")
    j.set_defaults(fn=cmd_judge)

    for p in (r, j):
        p.add_argument("--games", type=int, default=100,
                       help="paired games per pairing")
        p.add_argument("--comparison-games", type=int, default=1000,
                       help="paired games between the arms' final checkpoints")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--labels", default="all",
                       help="snapshot labels to rate, or 'all' for every one")
        p.add_argument("--fixed-draft", action="store_true",
                       help="evaluate the starter matchup instead of random drafts")

    l = sub.add_parser("ls", help="every run, newest first")
    l.add_argument("--limit", type=int, default=40)
    l.set_defaults(fn=cmd_ls)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
