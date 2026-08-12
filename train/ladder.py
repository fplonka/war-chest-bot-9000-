"""Evaluate a sparse checkpoint graph and turn the results into Elo.

    python train/ladder.py runs/mine --games 300
    python train/ladder.py runs/a runs/b --comparison-games 2000

Every saved checkpoint is included by default. Consecutive checkpoints within
each run form its learning curve; the first and final checkpoints play Greedy;
and each candidate final plays only its same-seed control final. Curve and
anchor edges use a modest fixed budget, while those explicit arm comparisons
use the larger budget. The graph is connected and has O(K) edges for K
checkpoints.

Games use paired seats and paired drafts. Each edge has a deterministic seed,
so rerunning the same graph reproduces the same evaluation schedule. Random
drafts are the default; `--fixed-draft` is the explicit exception.

Ratings come from the Bradley-Terry model, which is what Elo *is*: player `i`
beats `j` with probability `1 / (1 + 10 ** ((e_j - e_i) / 400))`. Draws count
half. Fitting is Zermelo's MM iteration, a handful of lines that cannot diverge
and needs no optimiser. The first reference player is pinned at 0, so which
references a ladder includes decides what its zero means -- with `--refs greedy`
the numbers read as Elo above Greedy, and are not comparable to a ladder that
also carried Random.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import truth
import warchest
from export_weights import load

# One drawn game added to every pairing before fitting. Without it a player who
# wins every game against everyone has infinite Elo, which is not a claim the
# data supports -- 60 wins from 60 games is consistent with anything above about
# +550. With it, an unbeaten player's rating is finite and reads as "at least
# this much", and a pairing with real losses in it is moved by well under 10
# Elo.
PRIOR = 1.0


def write_json(path, value):
    """Publish a complete result without exposing a partially written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=1)
    os.replace(tmp, path)


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


SEARCH = {"depth": 2, "iters": 64, "cfr": "linear", "warm": 0.0}


def parse_search(spec):
    """`"dcfr:64"` -> the SEARCH dict with that rule and iteration count."""
    if not spec:
        return None
    cfr, _, iters = spec.partition(":")
    return dict(SEARCH, cfr=cfr, iters=int(iters or SEARCH["iters"]))


def players_of(runs, refs=("greedy",), labels=None, search=None):
    """The ladder's entrants: the fixed bots, then every run's snapshots.

    The first reference is pinned at 0, so which references a ladder carries
    decides what its zero means. Greedy is the only reference worth playing.
    Random is not in this list and should not be added back: a game against an
    opponent 400 Elo away carries a tenth of the information of a game against
    an equal, and every trained checkpoint beats Random ~30-0. Those games are
    a foregone conclusion bought at full price. Keep the references fixed
    forever -- they are pure functions of the
    rules, which is what makes a rating from one ladder comparable to a rating
    from another. A pool of rotating champions was tried and removed: when the
    zero moves, old numbers stop meaning anything.

    A snapshot plays with the search settings it was *trained* with, read from
    the checkpoint. A net trained under one regret rule and rated under another
    is not the player the run produced, and before the settings were stored
    nothing downstream could tell.
    """
    ps = [{"name": r, "agent": r, "slot": 0, "t": None, "run": None,
           "search": SEARCH, "endpoint": False, "final": False, "order": -1,
           "experiment": None, "arm": None, "seed": None, "is_control": None}
          for r in refs]
    for run in runs:
        tag = os.path.basename(run.rstrip("/"))
        with open(f"{run}/log.json") as f:
            log = json.load(f)
        cfg = log.get("cfg", {})
        selected = [s for s in log.get("snapshots", [])
                    if labels is None or s["label"] in labels]
        for order, s in enumerate(selected):
            ck = torch.load(f"{run}/{s['file']}", map_location="cpu", weights_only=False)
            # A checkpoint older than a knob does not carry it, so fall back to
            # the run's config and then to the default.
            own = {k: cfg.get(k, v) for k, v in SEARCH.items()}
            own.update(ck.get("search") or {})
            ps.append({"name": f"{tag}.{s['label']}", "agent": "rebel",
                       "slot": len(ps), "t": s["t"], "file": s["file"],
                       "run": run, "search": search or own, "order": order,
                       "endpoint": order in (0, len(selected) - 1),
                       "final": s["label"] == "final",
                       "experiment": cfg.get("experiment"),
                       "arm": cfg.get("arm"), "seed": cfg.get("seed"),
                       "is_control": cfg.get("is_control")})
    return ps



def pairing_games(a, b, comparisons, curve_games, comparison_games):
    """Budget for one edge in the fixed linear comparison graph."""
    if a["run"] is None or b["run"] is None:
        p = b if a["run"] is None else a
        return curve_games if p["endpoint"] else 0
    if a["run"] == b["run"]:
        return curve_games if abs(a["order"] - b["order"]) == 1 else 0
    runs = frozenset((a["run"], b["run"]))
    return comparison_games if a["final"] and b["final"] and runs in comparisons else 0


def run(runs, out=None, games=60, temp=2.0, random_draft=True, seed=7,
        refs=("greedy",), labels=None, comparisons=(), comparison_games=0,
        gpu=False, search=None):
    """Evaluate the sparse comparison graph and fit Elo. Returns the ratings.

    Consecutive checkpoints form each learning curve, its endpoints anchor to
    Greedy, and explicit final-vs-control comparisons get the larger budget.
    The graph is linear in the number of checkpoints and connected.

    With `gpu=True` two solve services run (CUDA devices 0 and 1); each
    pairing loads side A's weights on device 0 and side B's on device 1, so
    both GPUs are busy while the ladder plays. All checkpoints must share
    one network shape (the services are compiled for it).
    """
    if out is None:
        out = runs[0]
    if games <= 0 or comparison_games < 0 or games % 2 or comparison_games % 2:
        raise ValueError("game budgets must be positive paired (even) counts")
    ps = players_of(runs, refs, labels, search)
    comparison_games = comparison_games or games
    comparisons = {frozenset(x) for x in comparisons}
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
    # Always the real game: the horizon's marker payoff is a training aid, and
    # scoring on it would rank whoever exploits it best.
    warchest.set_cap_value(0.0)

    k = len(ps)
    n = np.zeros((k, k))
    sc = np.zeros((k, k))
    pairs = []
    # Say what the fixed sparse graph will cost before spending it.
    plan = {(i, j): pairing_games(ps[i], ps[j], comparisons, games, comparison_games)
            for i in range(k) for j in range(i + 1, k)}
    played = sum(1 for v in plan.values() if v)
    total_games = sum(plan.values())
    print(f"[ladder] {k} players, {played} of {len(plan)} pairings, "
          f"{games} games per curve/anchor edge ({comparison_games} per comparison) "
          f"-> {total_games:,} games, about {total_games / 3 / 60:.0f} min",
          flush=True)
    for (i, j), n_ij in plan.items():
        # A seed per pairing rather than one shared across the tournament.
        # Within a pairing the two seatings are already paired on the same
        # stream, which is where the variance reduction is worth having; making
        # the pairings share a stream too would correlate their errors, and the
        # standard errors below assume they do not.
        a, b = ps[i], ps[j]
        if n_ij == 0:
            continue
        if gpu:
            if a["agent"] == "rebel":
                na = by_slot[a["slot"]]
                warchest.gpu_set_weights(na.dims, *na.flat(), device=0)
            if b["agent"] == "rebel":
                nb = by_slot[b["slot"]]
                warchest.gpu_set_weights(nb.dims, *nb.flat(), device=1)
        sa, sb = a["search"], b["search"]
        pair_seed = seed + 1000 * i + j
        kind = ("anchor" if a["run"] is None or b["run"] is None else
                "curve" if a["run"] == b["run"] else "comparison")
        play = lambda n, s: warchest.eval_match(
            n, s, a["agent"], b["agent"],
            depth=sa["depth"], iters=sa["iters"], cfr=sa["cfr"], warm=sa["warm"],
            temp=temp, slot_a=a["slot"], slot_b=b["slot"],
            random_draft=random_draft,
            depth_b=sb["depth"], iters_b=sb["iters"],
            cfr_b=sb["cfr"], warm_b=sb["warm"], gpu=gpu)

        w, l, d = play(n_ij, pair_seed)
        n[i][j] = n[j][i] = w + l + d
        sc[i][j], sc[j][i] = w + 0.5 * d, l + 0.5 * d
        pairs.append({"a": a["name"], "b": b["name"], "w": w, "l": l, "d": d,
                      "n": w + l + d, "budget": n_ij, "seed": pair_seed,
                      "kind": kind,
                      "score": round((w + 0.5 * d) / max(w + l + d, 1), 3)})
        print(f"  {a['name']:>28s} vs {b['name']:<28s} W{w:4d} L{l:4d} D{d:4d}  "
              f"score {pairs[-1]['score']:.3f}", flush=True)

    npr = n + PRIOR * (1 - np.eye(k)) * (n > 0)
    spr = sc + 0.5 * PRIOR * (1 - np.eye(k)) * (n > 0)
    elo = fit_elo(npr, spr)
    se = elo_stderr(n, elo)
    # The noise-free half of the same question. Elo says who wins; this says how
    # far the value head is from the fixed point of the solved operator, and the
    # two disagreeing is itself a result.
    terr = truth.errors([f"{p['run']}/{p['file']}" for p in nets])
    res = {"runs": list(runs), "curve_games": games,
           "comparison_games": comparison_games,
           "schedule_seed": seed,
           "draft_mode": "random" if random_draft else "fixed",
           "comparisons": sorted(sorted(x) for x in comparisons),
           "truth_set": truth.DEFAULT_SET if terr else None,
           "players": [{"name": p["name"], "run": p["run"], "t": p["t"],
                        "experiment": p["experiment"], "arm": p["arm"],
                        "seed": p["seed"], "is_control": p["is_control"],
                        "search": p["search"], "elo": round(float(e), 1),
                        "se": round(float(s), 1),
                        "truth": round(terr[f"{p['run']}/{p['file']}"][0], 6)
                                 if p.get("file") and terr else None,
                        "score": round(float(sc[i].sum() / max(n[i].sum(), 1)), 3)}
                       for i, (p, e, s) in enumerate(zip(ps, elo, se))],
           "pairs": pairs}
    write_json(f"{out}/ladder.json", res)

    zero = ps[0]["name"] if ps else "?"
    print(f"\n=== Elo ({out}, {zero} = 0) ===", flush=True)
    print(f"{'player':>28s} {'trained':>9s} {'elo':>7s} {'+-':>5s} {'score':>7s} "
          f"{'truth':>8s}", flush=True)
    for p in sorted(res["players"], key=lambda p: -p["elo"]):
        tm = f"{p['t'] / 60:.1f}min" if p["t"] is not None else "-"
        print(f"{p['name']:>28s} {tm:>9s} {p['elo']:>7.0f} {p['se']:>5.0f} "
              f"{p['score']:>7.3f} "
              f"{('%.5f' % p['truth']) if p['truth'] is not None else '-':>8s}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="one or more run directories")
    ap.add_argument("--out", dest="dest", default=None,
                    help="where to write ladder.json (default: the first run directory)")
    ap.add_argument("--games", type=int, default=100, help="games per curve/anchor edge")
    ap.add_argument("--comparison-games", type=int, default=0,
                    help="games for final-vs-control comparisons (default: --games)")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--fixed-draft", action="store_true",
                    help="evaluate the starter matchup instead of random drafts")
    ap.add_argument("--refs", default="greedy",
                    help="comma list of fixed reference bots, or empty for none. "
                         "The first is pinned at 0. Greedy is the only one worth "
                         "playing: see players_of.")
    ap.add_argument("--labels", default=None,
                    help="only these snapshot labels, comma-separated (default: all)")
    ap.add_argument("--seed", type=int, default=7)
    # Every checkpoint plays at the settings it trained with, which compares
    # whole systems. To compare the *networks*, give them all one search: an arm
    # trained at T=16 is otherwise charged for playing at T=16 as well.
    ap.add_argument("--eval-search", default="",
                    help="force one search on every checkpoint, e.g. dcfr:64")
    ap.add_argument("--gpu", action="store_true",
                    help="two solve services (CUDA 0/1), one per pairing side")
    args = ap.parse_args()
    comparisons = [(args.runs[0], r) for r in args.runs[1:]]
    run(args.runs, out=args.dest, games=args.games,
        comparisons=comparisons, comparison_games=args.comparison_games,
        temp=args.temp,
        random_draft=not args.fixed_draft,
        seed=args.seed,
        refs=tuple(x for x in args.refs.split(",") if x),
        labels=args.labels.split(",") if args.labels else None,
        search=parse_search(args.eval_search), gpu=args.gpu)


if __name__ == "__main__":
    main()
