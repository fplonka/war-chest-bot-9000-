"""Play snapshots against Greedy (and each other's finals) and turn that into Elo.

    python train/ladder.py runs/mine --games 100
    python train/ladder.py runs/a runs/b --games 100

Every checkpoint plays Greedy. That is the curve: strength against a fixed
bot, over training time. When several runs are named, their finals also play
each other. Random drafts are the default; `--fixed-draft` is the exception.

Greedy is pinned at 0. Ratings come from Bradley-Terry / Zermelo MM.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warchest
from export_weights import load

PRIOR = 1.0
SEARCH = {"depth": 2, "iters": 64, "cfr": "linear", "warm": 0.0}


def write_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=1)
    os.replace(tmp, path)


def fit_elo(n, w):
    """Bradley-Terry ratings by MM iteration (Zermelo 1929; Hunter 2004)."""
    k = len(n)
    p = np.ones(k)
    for _ in range(10_000):
        prev = p.copy()
        for i in range(k):
            den = sum(n[i][j] / (p[i] + p[j]) for j in range(k) if j != i)
            p[i] = w[i].sum() / den if den > 0 else p[i]
        p /= p.sum()
        if np.max(np.abs(np.log(p / prev))) < 1e-12:
            break
    return 400.0 * np.log10(p / p[0])


def elo_stderr(n, elo):
    c = 400.0 / math.log(10.0)
    out = []
    for i in range(len(n)):
        info = 0.0
        for j in range(len(n)):
            if i == j or n[i][j] == 0:
                continue
            q = 1.0 / (1.0 + 10.0 ** ((elo[j] - elo[i]) / 400.0))
            info += n[i][j] * q * (1 - q) / (c * c)
        out.append(float("inf") if info == 0 else 1.0 / math.sqrt(info))
    return out


def parse_search(spec):
    if not spec:
        return None
    cfr, _, iters = spec.partition(":")
    return dict(SEARCH, cfr=cfr, iters=int(iters or SEARCH["iters"]))


def players_of(runs, labels=None, search=None):
    ps = [{"name": "greedy", "agent": "greedy", "slot": 0, "t": None,
           "run": None, "search": SEARCH, "final": False}]
    for run in runs:
        tag = os.path.basename(run.rstrip("/"))
        with open(f"{run}/log.json") as f:
            log = json.load(f)
        cfg = log.get("cfg", {})
        selected = [s for s in log.get("snapshots", [])
                    if labels is None or s["label"] in labels]
        for s in selected:
            ck = torch.load(f"{run}/{s['file']}", map_location="cpu", weights_only=False)
            own = {k: cfg.get(k, v) for k, v in SEARCH.items()}
            own.update(ck.get("search") or {})
            ps.append({"name": f"{tag}.{s['label']}", "agent": "rebel",
                       "slot": len(ps), "t": s["t"], "file": s["file"],
                       "run": run, "search": search or own,
                       "final": s["label"] == "final"})
    return ps


def edges(ps):
    """Every net vs Greedy; finals vs finals when there is more than one."""
    greedy = next(i for i, p in enumerate(ps) if p["agent"] == "greedy")
    nets = [i for i, p in enumerate(ps) if p["agent"] == "rebel"]
    out = [(i, greedy) for i in nets]
    finals = [i for i in nets if ps[i]["final"]]
    out.extend((finals[a], finals[b])
               for a in range(len(finals)) for b in range(a + 1, len(finals)))
    return out


def run(runs, out=None, games=100, temp=2.0, random_draft=True, seed=7,
        labels=None, gpu=False, search=None):
    if out is None:
        out = runs[0]
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even count")
    ps = players_of(runs, labels, search)
    nets = [p for p in ps if p["agent"] == "rebel"]
    if not nets:
        raise SystemExit(f"{runs}: no snapshots in log.json")
    by_slot = {}
    for p in nets:
        net = load(f"{p['run']}/{p['file']}")
        net.push(p["slot"])
        by_slot[p["slot"]] = net
    if gpu:
        shapes = {tuple(n.dims) for n in by_slot.values()}
        if len(shapes) != 1:
            raise SystemExit(f"--gpu needs one shared network shape, got {shapes}")
        first = next(iter(by_slot.values()))
        warchest.gpu_start(first.dims, *first.flat(), devices=[0, 1])
    warchest.set_cap_value(0.0)

    plan = edges(ps)
    print(f"[ladder] {len(ps)} players, {len(plan)} pairings, {games} games "
          f"-> {len(plan) * games:,} games, about {len(plan) * games / 3 / 60:.0f} min",
          flush=True)

    k = len(ps)
    n = np.zeros((k, k))
    sc = np.zeros((k, k))
    pairs = []
    for i, j in plan:
        a, b = ps[i], ps[j]
        if gpu:
            if a["agent"] == "rebel":
                na = by_slot[a["slot"]]
                warchest.gpu_set_weights(na.dims, *na.flat(), device=0)
            if b["agent"] == "rebel":
                nb = by_slot[b["slot"]]
                warchest.gpu_set_weights(nb.dims, *nb.flat(), device=1)
        sa, sb = a["search"], b["search"]
        pair_seed = seed + 1000 * i + j
        w, l, d = warchest.eval_match(
            games, pair_seed, a["agent"], b["agent"],
            depth=sa["depth"], iters=sa["iters"], cfr=sa["cfr"], warm=sa["warm"],
            temp=temp, slot_a=a["slot"], slot_b=b["slot"],
            random_draft=random_draft,
            depth_b=sb["depth"], iters_b=sb["iters"], gpu=gpu)
        n[i][j] = n[j][i] = w + l + d
        sc[i][j], sc[j][i] = w + 0.5 * d, l + 0.5 * d
        pairs.append({"a": a["name"], "b": b["name"], "w": w, "l": l, "d": d,
                      "n": w + l + d, "score": round((w + 0.5 * d) / max(w + l + d, 1), 3)})
        print(f"  {a['name']:>28s} vs {b['name']:<28s} W{w:4d} L{l:4d} D{d:4d}  "
              f"score {pairs[-1]['score']:.3f}", flush=True)

    npr = n + PRIOR * (1 - np.eye(k)) * (n > 0)
    spr = sc + 0.5 * PRIOR * (1 - np.eye(k)) * (n > 0)
    elo = fit_elo(npr, spr)
    se = elo_stderr(n, elo)
    res = {"runs": list(runs), "games": games, "schedule_seed": seed,
           "draft_mode": "random" if random_draft else "fixed",
           "players": [{"name": p["name"], "run": p["run"], "t": p["t"],
                        "search": p["search"], "elo": round(float(e), 1),
                        "se": round(float(s), 1),
                        "score": round(float(sc[i].sum() / max(n[i].sum(), 1)), 3)}
                       for i, (p, e, s) in enumerate(zip(ps, elo, se))],
           "pairs": pairs}
    write_json(f"{out}/ladder.json", res)
    print(f"\n=== Elo ({out}, greedy = 0) ===", flush=True)
    print(f"{'player':>28s} {'trained':>9s} {'elo':>7s} {'+-':>5s} {'score':>7s}",
          flush=True)
    for p in sorted(res["players"], key=lambda p: -p["elo"]):
        tm = f"{p['t'] / 60:.1f}min" if p["t"] is not None else "-"
        print(f"{p['name']:>28s} {tm:>9s} {p['elo']:>7.0f} {p['se']:>5.0f} "
              f"{p['score']:>7.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", dest="dest", default=None)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--fixed-draft", action="store_true")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-search", default="")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    run(args.runs, out=args.dest, games=args.games, temp=args.temp,
        random_draft=not args.fixed_draft, seed=args.seed,
        labels=args.labels.split(",") if args.labels else None,
        search=parse_search(args.eval_search), gpu=args.gpu)


if __name__ == "__main__":
    main()
