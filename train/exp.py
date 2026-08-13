"""Run an experiment end to end, and list what has been run.

    python train/exp.py run sanity --seeds 1
    python train/exp.py ls
    python train/exp.py judge runs/sanity-base
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


def unique_dir(name):
    """`runs/name`, or `runs/name-2` if that directory already exists."""
    path = os.path.join(RUNS, name)
    if not os.path.exists(path):
        return name, path
    n = 2
    while os.path.exists(os.path.join(RUNS, f"{name}-{n}")):
        n += 1
    taken = f"{name}-{n}"
    print(f"  [exp] {name} exists; writing {taken}", flush=True)
    return taken, os.path.join(RUNS, taken)


def launch(cfg, out):
    os.makedirs(out, exist_ok=True)
    path = f"{out}/config.json"
    ladder.write_json(path, dataclasses.asdict(cfg))
    cmd = [sys.executable, f"{HERE}/train.py", "--config", path]
    print(f"\n=== {out} ===\n{' '.join(cmd)}", flush=True)
    t = time.time()
    with open(f"{out}/train.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"=== {out}: {'ok' if rc == 0 else f'FAILED rc={rc}'} "
          f"in {(time.time() - t) / 60:.1f} min ===", flush=True)
    if rc != 0:
        print(f"    tail -20 {out}/train.log", flush=True)
    return rc == 0


def judge(runs, games, gpu, seed=7, labels=None, fixed_draft=False):
    result = ladder.run(runs, out=runs[0], games=games, gpu=gpu, seed=seed,
                        labels=labels, random_draft=not fixed_draft)
    for r in runs:
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
            "the tree is dirty. Commit first, or pass --force if you mean it.")
    arms = config.arms(args.name)
    todo = [(f"{args.name}-{lab}" + (f"-s{s}" if args.seeds > 1 else ""),
             dataclasses.replace(cfg, seed=s, experiment=args.name, arm=lab))
            for s in range(1, args.seeds + 1)
            for lab, cfg in arms]
    print(f"[exp] {args.name} at {sha}: {len(todo)} runs "
          f"({len(arms)} arms x {args.seeds} seeds)")
    for name, cfg in todo:
        print(f"  {name:<28s} {config.delta(cfg) or 'baseline'}")
    if args.dry_run:
        return
    done = []
    for name, cfg in todo:
        name, out = unique_dir(name)
        cfg = dataclasses.replace(cfg, out=out,
                                  dump_buffer=f"{out}/buf.npz" if args.dump else "")
        if not launch(cfg, out):
            raise SystemExit(f"[exp] {out} failed; experiment aborted")
        done.append(out)
    judge(done, args.games, gpu=True, seed=args.seed,
          labels=None if args.labels == "all" else args.labels.split(","),
          fixed_draft=args.fixed_draft)


def cmd_judge(args):
    judge(args.runs, args.games, gpu=args.gpu, seed=args.seed,
          labels=None if args.labels == "all" else args.labels.split(","),
          fixed_draft=args.fixed_draft)


def cmd_ls(args):
    rows = []
    if not os.path.isdir(RUNS):
        return
    for d in sorted(os.listdir(RUNS)):
        try:
            with open(f"{RUNS}/{d}/log.json") as f:
                log = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(log, dict):
            continue
        cfg, eps = log.get("cfg", {}), log.get("epochs", [])
        elo = ""
        try:
            with open(f"{RUNS}/{d}/ladder.json") as f:
                lad = json.load(f)
            fin = next((p for p in lad["players"] if p["name"].endswith(".final")), None)
            if fin:
                elo = f"{fin['elo']:.0f}±{fin['se']:.0f}"
        except (OSError, json.JSONDecodeError, TypeError):
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
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--force", action="store_true")
    r.add_argument("--dump", action="store_true")
    r.set_defaults(fn=cmd_run)

    j = sub.add_parser("judge", help="rate finished runs")
    j.add_argument("runs", nargs="+")
    j.add_argument("--gpu", action="store_true")
    j.set_defaults(fn=cmd_judge)

    for p in (r, j):
        p.add_argument("--games", type=int, default=100)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--labels", default="all")
        p.add_argument("--fixed-draft", action="store_true")

    l = sub.add_parser("ls", help="every run, newest first")
    l.add_argument("--limit", type=int, default=40)
    l.set_defaults(fn=cmd_ls)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
