"""Play one or more runs' snapshots against each other and turn the results into Elo.

    python train/ladder.py runs/mine --games 300
    python train/ladder.py runs/a runs/b --focus a.final,b.final --focus-games 2000

With several run directories the snapshots are entered as `run.label` so the
runs' overlapping labels (`init`, `s1`, ...) do not collide, each run's
checkpoints go into their own slots, and Greedy appears once. Rating several
arms in one ladder is the point: every game places every player, so an arm is
compared to its control through the whole field rather than through one match.

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

Cheap is not a figure of speech: a game is on the order of a hundred solves and
generation runs above a thousand solves a second, so a 2,000-game pairing costs
a few minutes against the hour that produced the checkpoints. Ladders here used
to run 30 to 100 games, which resolves nothing finer than about 70 Elo -- and
the architecture experiments this project has already run were decided, wrongly,
inside that band.

Ratings come from the Bradley-Terry model, which is what Elo *is*: player `i`
beats `j` with probability `1 / (1 + 10 ** ((e_j - e_i) / 400))`. Draws count
half. Fitting is Zermelo's MM iteration, a handful of lines that cannot diverge
and needs no optimiser. The first reference player is pinned at 0, so which
references a ladder includes decides what its zero means -- with `--refs greedy`
the numbers read as Elo above Greedy, and are not comparable to a ladder that
also carried Random.
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


SEARCH = {"depth": 2, "iters": 64, "cfr": "linear", "warm": 0.0, "zero_sum": True}


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
    a foregone conclusion bought at full price. Keep the references fixed forever -- they are pure functions of the
    rules, which is what makes a rating from one ladder comparable to a rating
    from another. A pool of rotating champions was tried and removed: when the
    zero moves, old numbers stop meaning anything.

    A snapshot plays with the search settings it was *trained* with, read from
    the checkpoint. A net trained under one regret rule and rated under another
    is not the player the run produced, and before the settings were stored
    nothing downstream could tell.
    """
    ps = [{"name": r, "agent": r, "slot": 0, "t": None, "run": None,
           "search": SEARCH} for r in refs]
    for run in runs:
        tag = os.path.basename(run.rstrip("/"))
        with open(f"{run}/log.json") as f:
            log = json.load(f)
        cfg = log.get("cfg", {})
        for s in log.get("snapshots", []):
            if labels is not None and s["label"] not in labels:
                continue
            ck = torch.load(f"{run}/{s['file']}", map_location="cpu", weights_only=False)
            # A checkpoint older than a knob does not carry it, so fall back to
            # the run's config and then to the default.
            own = {k: cfg.get(k, v) for k, v in SEARCH.items()}
            own.update(ck.get("search") or {})
            ps.append({"name": f"{tag}.{s['label']}", "agent": "rebel",
                       "slot": len(ps), "t": s["t"], "file": s["file"],
                       "run": run, "search": search or own})
    return ps



# ---------------------------------------------------------------- SPRT
# Fixed sample sizes are the wrong shape for "is A better than B". Pick 2,000
# games in advance and you either spend them proving something that was obvious
# after 300, or spend them all and land inside the noise anyway.
#
# Wald's sequential probability ratio test asks the question the experiment
# actually asks: H0 says A is no better than B; H1 says A is better by at least
# ELO1. After each block of games the log-likelihood ratio moves, and the test
# stops the moment it crosses a bound. Computer chess has tested engine changes
# this way for years, at the same effect sizes we care about.
#
# The model is the normal approximation to the per-game score (0, 0.5, 1) with
# the variance estimated from the games themselves -- the generalized SPRT
# everyone actually runs. Draws need no special treatment: they are simply
# scores of 0.5, and the horizon here manufactures a lot of them, which shows up
# honestly as a smaller variance and therefore a faster decision.
ELO0, ELO1 = 0.0, 15.0      # indifference band: below 15 Elo we do not care

# Stopping early is what makes a full round robin affordable, and it costs
# something: a pairing stopped at its boundary has a win rate biased slightly
# towards that boundary, so a rating fitted from truncated pairings is not the
# rating a fixed schedule would give. The bias is small next to the effect sizes
# a learning curve shows, and it buys spending the games where the curve is
# genuinely uncertain rather than on pairings settled after 100 games.
ALPHA = BETA = 0.05


def expected_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def llr(w, l, d):
    """Log-likelihood ratio of H1 (elo=ELO1) against H0 (elo=ELO0)."""
    n = w + l + d
    if n < 2:
        return 0.0
    total = w + 0.5 * d
    mean = total / n
    # Second moment of the per-game score, then the sample variance.
    var = (w + 0.25 * d) / n - mean * mean
    if var <= 1e-9:                      # every game identical; no information
        return 0.0
    m0, m1 = expected_score(ELO0), expected_score(ELO1)
    return (m1 - m0) / var * (total - n * (m0 + m1) / 2.0)


def sprt_verdict(w, l, d):
    """`"H1"` (A is better), `"H0"` (it is not), or `None` for keep playing."""
    lo = math.log(BETA / (1.0 - ALPHA))
    hi = math.log((1.0 - BETA) / ALPHA)
    r = llr(w, l, d)
    return "H1" if r >= hi else ("H0" if r <= lo else None)


def pairing_games(a, b, focus, curve_games, focus_games):
    """How many games this pairing is worth. 0 skips it.

    Three kinds of pairing, and only two of them are worth paying for:

    * **Within one run** — the learning curve. Consecutive snapshots differ by
      far more than a few hundred games can miss, so this is cheap.
    * **Between two runs' finals** — the comparison the experiment exists to
      make. This is where the games go.
    * **Between one run's `s1` and another's `s2`** — nobody asked. Skipped.
      These are most of the pairings once several arms are in one ladder, and
      they are what turns a round robin quadratic in the snapshot count.

    Everything still reaches everything through Greedy and through its own run's
    chain, so the ratings stay on one connected graph and remain comparable.
    """
    if a["run"] is None or b["run"] is None:      # a fixed reference
        return curve_games
    if {a["name"], b["name"]} <= focus:
        return focus_games
    if a["run"] == b["run"]:
        return curve_games
    return 0


SPRT_BLOCK = 100            # games between tests; small enough to stop early


def run(runs, out=None, games=60, temp=2.0, random_draft=False, seed=7,
        refs=("greedy",), labels=None, focus=(), focus_games=0, gpu=False,
        sprt=True, search=None):
    """Round robin, Elo fit, `ladder.json`, printed table. Returns the ratings.

    Games go where the information is. A round robin at one sample size spends
    most of its games on pairings nobody asked about, and a pairing carries
    information roughly in proportion to its game count: 100 games resolve
    about 70 Elo, 1,000 about 22, 5,000 about 10. So `focus` names the players
    whose comparison the experiment is actually about -- normally the arms'
    final checkpoints -- and any pairing between two of them gets
    `focus_games` instead of `games`. The Bradley-Terry fit does not care that
    the counts differ.

    With `gpu=True` two solve services run (CUDA devices 0 and 1); each
    pairing loads side A's weights on device 0 and side B's on device 1, so
    both GPUs are busy while the ladder plays. All checkpoints must share
    one network shape (the services are compiled for it), and v1-era
    checkpoints cannot play on the GPU.
    """
    if out is None:
        out = runs[0]
    ps = players_of(runs, refs, labels, search)
    focus, focus_games = set(focus), focus_games or games
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
        if next(iter(shapes))[0] != 3:
            raise SystemExit("--gpu cannot play v1-era checkpoints")
        first = next(iter(by_slot.values()))
        warchest.gpu_start(first.dims, *first.flat(), devices=[0, 1])
    # Always the real game: the horizon's marker payoff is a training aid, and
    # scoring on it would rank whoever exploits it best.
    warchest.set_cap_value(0.0)

    k = len(ps)
    n = np.zeros((k, k))
    sc = np.zeros((k, k))
    pairs = []
    # Pairings grow with the square of the player count, so say what this will
    # cost before spending it. A game is on the order of a hundred solves, and
    # the golden run generates ~1,300 solves a second.
    plan = {(i, j): pairing_games(ps[i], ps[j], focus, games, focus_games)
            for i, j in itertools.combinations(range(k), 2)}
    played = sum(1 for v in plan.values() if v)
    total_games = sum(plan.values())
    # With SPRT these counts are ceilings, not schedules: a settled pairing
    # stops at the first block. Net-against-net games run about 3/s on the two
    # 3090s, and a game against Greedy is far cheaper.
    print(f"[ladder] {k} players, {played} of {len(plan)} pairings, "
          f"up to {games} games each ({focus_games} between the arms' finals) "
          f"-> at most {total_games:,} games, {total_games / 3 / 60:.0f} min",
          flush=True)
    for i, j in itertools.combinations(range(k), 2):
        # A seed per pairing rather than one shared across the tournament.
        # Within a pairing the two seatings are already paired on the same
        # stream, which is where the variance reduction is worth having; making
        # the pairings share a stream too would correlate their errors, and the
        # standard errors below assume they do not.
        a, b = ps[i], ps[j]
        if gpu:
            if a["agent"] == "rebel":
                na = by_slot[a["slot"]]
                warchest.gpu_set_weights(na.dims, *na.flat(), device=0)
            if b["agent"] == "rebel":
                nb = by_slot[b["slot"]]
                warchest.gpu_set_weights(nb.dims, *nb.flat(), device=1)
        sa, sb = a["search"], b["search"]
        n_ij = plan[(i, j)]
        if n_ij == 0:
            continue
        play = lambda n, s: warchest.eval_match(
            n, s, a["agent"], b["agent"],
            depth=sa["depth"], iters=sa["iters"], cfr=sa["cfr"], warm=sa["warm"],
            zero_sum=sa["zero_sum"],
            temp=temp, slot_a=a["slot"], slot_b=b["slot"],
            random_draft=random_draft,
            depth_b=sb["depth"], iters_b=sb["iters"],
            cfr_b=sb["cfr"], warm_b=sb["warm"], zero_sum_b=sb["zero_sum"], gpu=gpu)

        verdict = None
        if sprt:
            # Every pairing plays in blocks and stops as soon as the evidence
            # is conclusive, or at n_ij if it never is. A lopsided pairing is
            # settled by the first block; the games saved there are spent on
            # the neighbouring snapshots, which is where the curve is actually
            # uncertain.
            w = l = d = 0
            for blk in range(0, n_ij, SPRT_BLOCK):
                bw, bl, bd = play(min(SPRT_BLOCK, n_ij - blk),
                                  seed + 1000 * i + j + 7919 * blk)
                w, l, d = w + bw, l + bl, d + bd
                verdict = sprt_verdict(w, l, d)
                if verdict:
                    break
        else:
            w, l, d = play(n_ij, seed + 1000 * i + j)
        n[i][j] = n[j][i] = w + l + d
        sc[i][j], sc[j][i] = w + 0.5 * d, l + 0.5 * d
        pairs.append({"a": a["name"], "b": b["name"], "w": w, "l": l, "d": d,
                      "n": w + l + d, "planned": n_ij, "sprt": verdict,
                      "llr": round(llr(w, l, d), 2),
                      "score": round((w + 0.5 * d) / max(w + l + d, 1), 3)})
        note = {"H1": f"  SPRT: better by >{ELO1:.0f} Elo",
                "H0": f"  SPRT: not better by {ELO1:.0f} Elo",
                None: "  (inconclusive)"}[verdict] if sprt else ""
        print(f"  {a['name']:>28s} vs {b['name']:<28s} W{w:4d} L{l:4d} D{d:4d}  "
              f"score {pairs[-1]['score']:.3f}{note}", flush=True)

    npr = n + PRIOR * (1 - np.eye(k)) * (n > 0)
    spr = sc + 0.5 * PRIOR * (1 - np.eye(k)) * (n > 0)
    elo = fit_elo(npr, spr)
    se = elo_stderr(n, elo)
    # The noise-free half of the same question. Elo says who wins; this says how
    # far the value head is from the fixed point of the solved operator, and the
    # two disagreeing is itself a result.
    terr = truth.errors([f"{p['run']}/{p['file']}" for p in nets])
    res = {"runs": list(runs), "games_per_pair": games, "focus_games": focus_games,
           "focus": sorted(focus), "truth_set": truth.DEFAULT_SET if terr else None,
           "players": [{"name": p["name"], "t": p["t"], "search": p["search"], "elo": round(float(e), 1),
                        "se": round(float(s), 1),
                        "truth": round(terr[f"{p['run']}/{p['file']}"][0], 6)
                                 if p.get("file") and terr else None,
                        "score": round(float(sc[i].sum() / max(n[i].sum(), 1)), 3)}
                       for i, (p, e, s) in enumerate(zip(ps, elo, se))],
           "pairs": pairs}
    with open(f"{out}/ladder.json", "w") as f:
        json.dump(res, f, indent=1)

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
    ap.add_argument("--games", type=int, default=300, help="paired games per pairing")
    ap.add_argument("--focus", default="",
                    help="comma list of players whose pairings get --focus-games")
    ap.add_argument("--focus-games", type=int, default=0,
                    help="paired games between two focus players (default: --games)")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--random-draft", action="store_true")
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
    cfgs = [json.load(open(f"{d}/log.json")).get("cfg", {}) for d in args.runs]
    run(args.runs, out=args.dest, games=args.games,
        focus=[x for x in args.focus.split(",") if x], focus_games=args.focus_games,
        temp=args.temp,
        random_draft=args.random_draft or any(c.get("random_draft", False) for c in cfgs),
        seed=args.seed,
        refs=tuple(x for x in args.refs.split(",") if x),
        labels=args.labels.split(",") if args.labels else None,
        search=parse_search(args.eval_search), gpu=args.gpu)


if __name__ == "__main__":
    main()
