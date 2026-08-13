"""Rate snapshots by playing the informative pairs.

    python train/ladder.py runs/mine --games 40
    python train/ladder.py runs/a runs/b --games 40

A game against a much weaker opponent is almost no information (Fisher
p(1-p) peaks at a coin flip). Consecutive snapshots start close, so they
are the first pairings. The rest of the budget goes to whoever is closest
in Elo right now — Swiss pairing, the same rule TrueSkill and Chatbot Arena
use — and a pair that has already gone 80–20 is done.

Greedy is pinned at 0. Random drafts.
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
BATCH = 20  # even; eval_match plays this many as colour-swapped pairs


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


def players_of(runs):
    ps = [{"name": "greedy", "agent": "greedy", "slot": 0, "t": None,
           "run": None, "search": SEARCH, "final": False}]
    for run in runs:
        tag = os.path.basename(run.rstrip("/"))
        with open(f"{run}/log.json") as f:
            log = json.load(f)
        cfg = log.get("cfg", {})
        for s in log.get("snapshots", []):
            ck = torch.load(f"{run}/{s['file']}", map_location="cpu", weights_only=False)
            own = {k: cfg.get(k, v) for k, v in SEARCH.items()}
            own.update(ck.get("search") or {})
            ps.append({"name": f"{tag}.{s['label']}", "agent": "rebel",
                       "slot": len(ps), "t": s["t"], "file": s["file"],
                       "run": run, "search": own,
                       "final": s["label"] == "final"})
    return ps


def chain(ps):
    """Greedy vs each run's first snapshot, then consecutive snapshots, then
    finals across runs. The prior that nearby-in-time nets are close."""
    greedy = next(i for i, p in enumerate(ps) if p["agent"] == "greedy")
    by_run = {}
    for i, p in enumerate(ps):
        if p["run"] is not None:
            by_run.setdefault(p["run"], []).append(i)
    out = []
    for idxs in by_run.values():
        idxs.sort(key=lambda i: ps[i]["t"] or 0)
        out.append((greedy, idxs[0]))
        out.extend(zip(idxs, idxs[1:]))
    finals = [i for i, p in enumerate(ps) if p.get("final")]
    out.extend((finals[a], finals[b])
               for a in range(len(finals)) for b in range(a + 1, len(finals)))
    seen, uniq = set(), []
    for i, j in out:
        key = (min(i, j), max(i, j))
        if i != j and key not in seen:
            seen.add(key)
            uniq.append((i, j))
    return uniq


def frozen(w, l, d):
    n = w + l + d
    if n < BATCH:
        return False
    s = (w + 0.5 * d) / n
    return min(s, 1.0 - s) < 0.20


def run(runs, games=40, gpu=False, seed=7):
    out = runs[0]
    if games < 2 or games % 2:
        raise ValueError("games must be a positive even count")
    ps = players_of(runs)
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

    k = len(ps)
    rec = {(i, j): [0, 0, 0] for i in range(k) for j in range(i + 1, k)}
    budget = games * len(nets)
    cap = games

    def elo_now():
        n = np.zeros((k, k))
        sc = np.zeros((k, k))
        for (i, j), (w, l, d) in rec.items():
            if w + l + d == 0:
                continue
            n[i][j] = n[j][i] = w + l + d
            sc[i][j], sc[j][i] = w + 0.5 * d, l + 0.5 * d
        npr = n + PRIOR * (1 - np.eye(k)) * (n > 0)
        spr = sc + 0.5 * PRIOR * (1 - np.eye(k)) * (n > 0)
        if n.sum() == 0:
            return np.zeros(k), n, sc
        return fit_elo(npr, spr), n, sc

    def play(i, j, n, tag):
        n -= n % 2
        if n < 2:
            return 0
        a, b = ps[i], ps[j]
        if gpu:
            if a["agent"] == "rebel":
                na = by_slot[a["slot"]]
                warchest.gpu_set_weights(na.dims, *na.flat(), device=0)
            if b["agent"] == "rebel":
                nb = by_slot[b["slot"]]
                warchest.gpu_set_weights(nb.dims, *nb.flat(), device=1)
        sa, sb = a["search"], b["search"]
        already = sum(rec[min(i, j), max(i, j)])
        w, l, d = warchest.eval_match(
            n, seed + 1009 * i + 101 * j + already, a["agent"], b["agent"],
            depth=sa["depth"], iters=sa["iters"], cfr=sa["cfr"], warm=sa["warm"],
            temp=2.0, slot_a=a["slot"], slot_b=b["slot"], random_draft=True,
            depth_b=sb["depth"], iters_b=sb["iters"], gpu=gpu)
        key = (min(i, j), max(i, j))
        if i > j:
            w, l = l, w
        rec[key][0] += w
        rec[key][1] += l
        rec[key][2] += d
        tw, tl, td = rec[key]
        s = (tw + 0.5 * td) / (tw + tl + td)
        print(f"  {tag:5s} {ps[key[0]]['name']:>24s} vs {ps[key[1]]['name']:<24s} "
              f"W{w:3d} L{l:3d} D{d:3d}  n={tw + tl + td:3d}  score {s:.3f}"
              f"{'  stop' if frozen(tw, tl, td) else ''}", flush=True)
        return n

    print(f"[ladder] {k} players, {budget} games ({BATCH}/batch, cap {cap}/pair)",
          flush=True)
    spent = 0
    for i, j in chain(ps):
        spent += play(i, j, min(BATCH, budget - spent, cap), "seed")
        if spent >= budget:
            break

    while spent + 2 <= budget:
        elo, _, _ = elo_now()
        best, best_h = None, -1.0
        for i in range(k):
            for j in range(i + 1, k):
                w, l, d = rec[i, j]
                n = w + l + d
                if n >= cap or frozen(w, l, d):
                    continue
                p = 1.0 / (1.0 + 10.0 ** ((elo[j] - elo[i]) / 400.0))
                h = p * (1.0 - p)
                if h > best_h:
                    best, best_h = (i, j), h
        if best is None:
            break
        n = min(BATCH, budget - spent, cap - sum(rec[best]))
        spent += play(best[0], best[1], n, "swiss")

    elo, nmat, sc = elo_now()
    se = elo_stderr(nmat, elo)
    pairs = [{"a": ps[i]["name"], "b": ps[j]["name"], "w": w, "l": l, "d": d,
              "n": w + l + d, "score": round((w + 0.5 * d) / max(w + l + d, 1), 3)}
             for (i, j), (w, l, d) in rec.items() if w + l + d]
    res = {"runs": list(runs), "games": games, "schedule_seed": seed,
           "draft_mode": "random",
           "players": [{"name": p["name"], "run": p["run"], "t": p["t"],
                        "search": p["search"], "elo": round(float(e), 1),
                        "se": round(float(s), 1),
                        "score": round(float(sc[i].sum() / max(nmat[i].sum(), 1)), 3)}
                       for i, (p, e, s) in enumerate(zip(ps, elo, se))],
           "pairs": pairs}
    write_json(f"{out}/ladder.json", res)
    print(f"\n=== Elo ({out}, greedy = 0) ===", flush=True)
    print(f"{'player':>28s} {'trained':>9s} {'elo':>7s} {'95%':>5s} {'score':>7s}",
          flush=True)
    for p in sorted(res["players"], key=lambda p: -p["elo"]):
        tm = f"{p['t'] / 60:.1f}min" if p["t"] is not None else "-"
        ci = 1.96 * p["se"] if math.isfinite(p["se"]) else float("inf")
        print(f"{p['name']:>28s} {tm:>9s} {p['elo']:>7.0f} {ci:>5.0f} "
              f"{p['score']:>7.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    run(args.runs, games=args.games, gpu=args.gpu)


if __name__ == "__main__":
    main()
