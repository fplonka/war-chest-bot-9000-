"""Play one or more runs' snapshots against each other and turn the results into Elo.

    python train/ladder.py runs/mine --games 60
    python train/ladder.py runs/a runs/b --games 100 --iters 64

With several run directories the snapshots are entered as `run.label` so the
runs' overlapping labels (`init`, `s1`, ...) do not collide, each run's
checkpoints go into their own slots, and Random and Greedy appear once.

A training run saves the network every few minutes and does not judge the
snapshots while it is running. This plays a full round robin between them --
plus Greedy (the handcrafted one-ply bot) and Random -- and fits one rating per
player, so a run's output is a curve of strength against training time rather
than a single number of unknown provenance.

Why this replaces gating. A mid-run match against a moving champion has a
standard error near +-0.05 at any affordable number of games, which is larger
than the improvement between two snapshots twenty minutes apart; promoting on it
is mostly promoting noise, and it spends training time to do so. Elo from a
round robin uses every game against every opponent to place every player, needs
no threshold, and is measured on the finished run where the games are cheap.

Ratings come from the Bradley-Terry model, which is what Elo *is*: player `i`
beats `j` with probability `1 / (1 + 10 ** ((e_j - e_i) / 400))`. Draws count
half. Fitting is Zermelo's MM iteration, a handful of lines that cannot diverge
and needs no optimiser. Random is pinned at 0 so the numbers mean the same thing
across runs.
"""

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warchest
from export_weights import load

# One drawn game added to every pairing before fitting. Without it a player who
# wins every game against everyone has infinite Elo, which is not a claim the
# data supports -- 60 wins from 60 games is consistent with anything above about
# +550. With it, an unbeaten player's rating is finite and reads as "at least
# this much", and a pairing with real losses in it is moved by well under 10
# Elo.
PRIOR = 1.0


def fit_elo(n, w):
    """Bradley-Terry ratings by MM iteration (Zermelo 1929; Hunter 2004).

    `n[i][j]` is games played and `w[i][j]` is `i`'s score in them (wins plus
    half the draws). Returns Elo, shifted so player 0 is at zero.
    """
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
    """Rough standard error per rating, from the Fisher information diagonal.

    Each game against an opponent of similar strength carries the most
    information; a game against someone 400 Elo away carries a tenth as much.
    The diagonal ignores the shared uncertainty between players, so read these
    as "how well is this player placed", not as a test between two of them.
    """
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


def players_of(runs, refs=True, labels=None, pool=None):
    """The ladder's entrants: optionally Random and Greedy, then each run's
    snapshots, then the pool file's entries.

    Random first because it is the fixed zero of the scale; Greedy second
    because it is the other fixed reference. Both are pure functions of the
    rules, so a rating measured here is comparable to one measured in any other
    run (`--no-refs` skips them). Every run's snapshots follow, named
    `run.label` and loaded into their own slots, so a combined ladder over
    several runs keeps each checkpoint's provenance. The pool file (see
    `runs/pool.json`) lists the best snapshots we have so far by explicit
    file, so a gate ladder is `ladder.py <newrun> --pool runs/pool.json`.
    """
    ps = []
    if refs:
        ps = [{"name": "random", "agent": "random", "slot": 0, "t": None, "run": None},
              {"name": "greedy", "agent": "greedy", "slot": 0, "t": None, "run": None}]
    for run in runs:
        tag = os.path.basename(run.rstrip("/"))
        with open(f"{run}/log.json") as f:
            log = json.load(f)
        for s in log.get("snapshots", []):
            if labels is not None and s["label"] not in labels:
                continue
            ps.append({"name": f"{tag}.{s['label']}", "agent": "rebel",
                       "slot": len(ps), "t": s["t"], "file": s["file"],
                       "run": run})
    for e in pool or []:
        ps.append({"name": e["name"], "agent": "rebel", "slot": len(ps),
                   "t": e.get("t"), "file": e["file"], "run": e["run"]})
    return ps


def run(runs, out=None, games=60, depth=2, iters=64, temp=2.0,
        random_draft=False, seed=7, refs=True, labels=None, pool=None,
        depth_b=0, iters_b=0):
    """Round robin, Elo fit, `ladder.json`, printed table. Returns the ratings."""
    if out is None:
        out = runs[0]
    ps = players_of(runs, refs, labels, pool)
    nets = [p for p in ps if p["agent"] == "rebel"]
    if not nets:
        raise SystemExit(f"{runs}: no snapshots in log.json")
    for p in nets:
        load(f"{p['run']}/{p['file']}").push(p["slot"])
    # Always the real game: the horizon's marker payoff is a training aid, and
    # scoring on it would rank whoever exploits it best.
    warchest.set_cap_value(0.0)

    k = len(ps)
    n = np.zeros((k, k))
    sc = np.zeros((k, k))
    pairs = []
    print(f"[ladder] {k} players, {k * (k - 1) // 2} pairings, {games} paired games each "
          f"(depth={depth} iters={iters})", flush=True)
    for i, j in itertools.combinations(range(k), 2):
        # A seed per pairing rather than one shared across the tournament.
        # Within a pairing the two seatings are already paired on the same
        # stream, which is where the variance reduction is worth having; making
        # the pairings share a stream too would correlate their errors, and the
        # standard errors below assume they do not.
        a, b = ps[i], ps[j]
        w, l, d = warchest.eval_match(games, seed + 1000 * i + j, a["agent"], b["agent"],
                                      depth=depth, iters=iters, temp=temp,
                                      slot_a=a["slot"], slot_b=b["slot"],
                                      random_draft=random_draft,
                                      depth_b=depth_b if depth_b > 0 else None,
                                      iters_b=iters_b if iters_b > 0 else None)
        n[i][j] = n[j][i] = w + l + d
        sc[i][j], sc[j][i] = w + 0.5 * d, l + 0.5 * d
        pairs.append({"a": a["name"], "b": b["name"], "w": w, "l": l, "d": d,
                      "score": round((w + 0.5 * d) / max(w + l + d, 1), 3)})
        print(f"  {a['name']:>28s} vs {b['name']:<28s} W{w:4d} L{l:4d} D{d:4d}  "
              f"score {pairs[-1]['score']:.3f}", flush=True)

    npr = n + PRIOR * (1 - np.eye(k)) * (n > 0)
    spr = sc + 0.5 * PRIOR * (1 - np.eye(k)) * (n > 0)
    elo = fit_elo(npr, spr)
    se = elo_stderr(n, elo)
    res = {"runs": list(runs), "games_per_pair": games, "depth": depth,
           "iters": iters, "depth_b": depth_b, "iters_b": iters_b,
           "players": [{"name": p["name"], "t": p["t"], "elo": round(float(e), 1),
                        "se": round(float(s), 1),
                        "score": round(float(sc[i].sum() / max(n[i].sum(), 1)), 3)}
                       for i, (p, e, s) in enumerate(zip(ps, elo, se))],
           "pairs": pairs}
    with open(f"{out}/ladder.json", "w") as f:
        json.dump(res, f, indent=1)

    print(f"\n=== Elo ({out}, random = 0) ===", flush=True)
    print(f"{'player':>28s} {'trained':>9s} {'elo':>7s} {'+-':>5s} {'score':>7s}", flush=True)
    for p in sorted(res["players"], key=lambda p: -p["elo"]):
        tm = f"{p['t'] / 60:.1f}min" if p["t"] is not None else "-"
        print(f"{p['name']:>28s} {tm:>9s} {p['elo']:>7.0f} {p['se']:>5.0f} "
              f"{p['score']:>7.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="+", help="one or more run directories")
    ap.add_argument("--out", dest="dest", default=None,
                    help="where to write ladder.json (default: the first run directory)")
    ap.add_argument("--games", type=int, default=60,
                    help="paired games per pairing")
    ap.add_argument("--depth", type=int, default=-1)
    ap.add_argument("--iters", type=int, default=-1)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--random-draft", action="store_true")
    ap.add_argument("--no-refs", action="store_true",
                    help="skip the Random and Greedy references")
    ap.add_argument("--labels", default=None,
                    help="only these snapshot labels, comma-separated (default: all)")
    ap.add_argument("--pool", default=None,
                    help="json file of fixed snapshot entries (the best so far)")
    ap.add_argument("--depth-b", type=int, default=0,
                    help="side B's search depth (default: same as side A)")
    ap.add_argument("--iters-b", type=int, default=0,
                    help="side B's CFR iterations (default: same as side A)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    # Play at the runs' own search settings unless told otherwise: a checkpoint
    # trained at one iteration count and rated at another is a different agent.
    # With several runs the defaults are the strongest settings any of them
    # used, so no run is rated below the search it trained with.
    cfgs = [json.load(open(f"{d}/log.json")).get("cfg", {}) for d in args.out]
    depth = args.depth if args.depth > 0 else max(c.get("depth", 2) for c in cfgs)
    iters = args.iters if args.iters > 0 else max(c.get("iters", 64) for c in cfgs)
    pool = json.load(open(args.pool)).get("entries", []) if args.pool else None
    run(args.out, out=args.dest, games=args.games, depth=depth, iters=iters,
        temp=args.temp,
        random_draft=args.random_draft or any(c.get("random_draft", False)
                                              for c in cfgs),
        seed=args.seed, refs=not args.no_refs,
        labels=args.labels.split(",") if args.labels else None,
        pool=pool, depth_b=args.depth_b, iters_b=args.iters_b)


if __name__ == "__main__":
    main()
