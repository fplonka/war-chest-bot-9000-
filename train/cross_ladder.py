#!/usr/bin/env python3
"""Play checkpoints from incompatible engine revisions in one exact ladder.

Each side runs in its own process against its own build, and every state is
cross-checked between them, so a disagreement is an error rather than a result.

`--concurrent` colour-swapped pairs are in flight at once. Each engine holds
that many live games and steps them in batches, and a relay solve releases the
GIL while it waits on the GPU service, which merges everything waiting into one
wave. Games are independent and carry their own RNG, so the points are exactly
the ones a serial ladder produces -- concurrency buys wall clock, not variance.
"""

import argparse
import collections
import json
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DRAFT_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 52, 53, 54]


def worker(checkpoint, train_root, device):
    """One engine process: a table of live relay games, stepped in batches.

    Every game in a batch takes its step on its own thread. A relay solve
    blocks inside the GPU service, which releases the GIL and merges whatever
    else is waiting into one wave — so a batch of a hundred games costs about
    what one game used to."""
    sys.path.insert(0, train_root)
    import warchest
    from export_weights import load

    net = load(checkpoint)
    net.push(0)
    w, b, ln = net.flat()
    warchest.gpu_start(net.dims, w, b, ln, devices=[int(device)])
    warchest.set_cap_value(0.0)
    games = {}
    pool = ThreadPoolExecutor(max_workers=512)

    def step(op, item):
        gid = item["id"]
        if op == "new":
            games[gid] = warchest.LiveGame(
                item["draft"], item["seat"], 0, item["depth"], item["iters"],
                item["seed"], True)
            return {"id": gid, "state": games[gid].snapshot()}
        if op == "drop":
            games.pop(gid, None)
            return {"id": gid}
        game = games[gid]
        if op == "chance":
            return {"id": gid, "state": game.relay_chance()}
        if op == "move":
            move = game.relay_agent_move()
            return {"id": gid, "move": move, "state": game.snapshot()}
        if op == "apply":
            move = item["move"]
            action = {"code": move["code"]}
            belief = [tuple(row) for row in move["belief"]]
            return {"id": gid, "state": game.relay_apply(action, belief)}
        raise ValueError(f"unknown operation {op}")

    for line in sys.stdin:
        try:
            req = json.loads(line)
            if req["op"] == "stop":
                break
            op = req["op"]
            items = list(pool.map(lambda x: step(op, x), req["items"]))
            result = {"items": items}
        except Exception as exc:
            result = {"error": repr(exc)}
        print("@@" + json.dumps(result, separators=(",", ":")), flush=True)


class Engine:
    def __init__(self, name, python, script, checkpoint, train_root, device, env=None):
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        self.name = name
        self.proc = subprocess.Popen(
            [python, script, "--worker", checkpoint, train_root, str(device)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env=child_env, bufsize=1)

    def call(self, op, items):
        """One batched step. Returns the per-game results keyed by game id."""
        if not items:
            return {}
        self.proc.stdin.write(
            json.dumps({"op": op, "items": items}, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name} worker exited with {self.proc.returncode}")
            if not line.startswith("@@"):
                print(f"[{self.name}] {line.rstrip()}", flush=True)
                continue
            result = json.loads(line[2:])
            if "error" in result:
                raise RuntimeError(f"{self.name}: {result['error']}")
            return {item["id"]: item for item in result["items"]}

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write('{"op":"stop"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)


def comparable(state):
    return {k: v for k, v in state.items()
            if k not in {"agent", "human", "log", "actions"}}


def assert_same(a, b, gid=None):
    left, right = comparable(a), comparable(b)
    if left != right:
        raise RuntimeError(
            f"engine states diverged (game {gid}):\n"
            + json.dumps({"new": left, "old": right}, indent=2))


def seats(gid):
    """`gid` is `2 * pair + new_seat`, so the game id carries its own colour."""
    return gid & 1


def advance(new, old, live, states):
    """Step every live game once, in as few batched round trips as the games'
    positions allow. `states` is the shared view both engines agree on."""
    chance = [{"id": g} for g in live if states[g]["is_chance"]]
    if chance:
        left = new.call("chance", chance)
        right = old.call("chance", chance)
        for item in chance:
            gid = item["id"]
            assert_same(left[gid]["state"], right[gid]["state"], gid)
            states[gid] = left[gid]["state"]

    # A game's mover is decided by its own colour, so each engine gets the half
    # of the batch it is to act in.
    for actor, peer in ((new, old), (old, new)):
        mine = [g for g in live
                if not states[g]["terminal"] and not states[g]["is_chance"]
                and (states[g]["to_act"] == seats(g)) == (actor is new)]
        if not mine:
            continue
        moved = actor.call("move", [{"id": g} for g in mine])
        applied = peer.call(
            "apply", [{"id": g, "move": moved[g]["move"]} for g in mine])
        for gid in mine:
            assert_same(moved[gid]["state"], applied[gid]["state"], gid)
            states[gid] = moved[gid]["state"]


def outcome(state, new_seat):
    winner = state["winner"]
    if winner < 0:
        return 0.5
    return 1.0 if winner == new_seat else 0.0


def binomial_tail(k, n):
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def summarize(points, pairs):
    wins = sum(x == 1.0 for x in points)
    draws = sum(x == 0.5 for x in points)
    losses = len(points) - wins - draws
    score = sum(points) / len(points) if points else 0.5
    decisive = wins + losses
    game_p = min(1.0, 2 * binomial_tail(max(wins, losses), decisive))
    pair_wins = sum(x > 1.0 for x in pairs)
    pair_losses = sum(x < 1.0 for x in pairs)
    pair_p = min(
        1.0,
        2 * binomial_tail(max(pair_wins, pair_losses), pair_wins + pair_losses),
    )
    if pair_wins > pair_losses and pair_p < 0.05:
        verdict = "new"
    elif pair_losses > pair_wins and pair_p < 0.05:
        verdict = "old"
    else:
        verdict = "inconclusive"
    elo = float("inf") if score == 1 else float("-inf") if score == 0 else \
        400 * math.log10(score / (1 - score))
    return {
        "wins": wins, "losses": losses, "draws": draws,
        "score": round(score, 4), "elo": round(elo, 1),
        "decisive_two_sided_p": float(f"{game_p:.6g}"),
        "paired_wins": pair_wins, "paired_losses": pair_losses,
        "paired_ties": len(pairs) - pair_wins - pair_losses,
        "paired_two_sided_p": float(f"{pair_p:.6g}"),
        "superior": verdict,
    }

def result_record(args, points, pairs, complete):
    return {
        "new": args.new,
        "old": args.old,
        "games": args.games,
        "seed": args.seed,
        "depth": args.depth,
        "iters": args.iters,
        "paired_random_drafts": True,
        "complete": complete,
        "points": points,
        "pair_points": pairs,
        "result": summarize(points, pairs),
    }


def write_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def pair_drafts(seed, npairs):
    """Every pair's draft, in the order one sequential RNG produces them. Kept
    separate from playing so the pairs are the same whoever plays them, and a
    resumed or re-sharded ladder reproduces the ones already on disk."""
    rng = random.Random(seed)
    out = []
    for _ in range(npairs):
        units = rng.sample(DRAFT_POOL, 8)
        out.append({
            "white_units": units[:4],
            "black_units": units[4:],
            "first_player": "white" if rng.randrange(2) == 0 else "black",
        })
    return out


def controller(args):
    if args.games < 2 or args.games % 2:
        raise ValueError("--games must be a positive even number")
    out = Path(args.out)
    points, pairs = [], []
    if out.exists():
        previous = json.loads(out.read_text())
        identity = ("new", "old", "games", "seed", "depth", "iters")
        expected = result_record(args, [], [], False)
        if any(previous.get(key) != expected[key] for key in identity):
            raise ValueError(f"{out} belongs to a different ladder")
        points = previous.get("points", [])
        pairs = previous.get("pair_points", [])
        if len(points) != 2 * len(pairs):
            raise ValueError(f"{out} ends inside a colour-swapped pair")
        if previous.get("complete"):
            print(json.dumps(previous, indent=2))
            return

    npairs = args.games // 2
    drafts = pair_drafts(args.seed, npairs)
    queue = collections.deque(range(len(pairs), npairs))
    finished = {}
    script = str(Path(__file__).resolve())
    new = Engine("new", args.python, script, args.new,
                 args.new_train, args.new_device)
    old = Engine("old", args.python, script, args.old, args.old_train,
                 args.old_device, {"PYTHONPATH": args.old_python})
    # Each game carries its own RNG and its own tree, so playing `concurrent`
    # of them at once changes nothing about any single game -- the points are
    # the ones a serial ladder would have produced. It only means the device
    # sees a wave of solves instead of one.
    live, states = [], {}
    started = time.time()
    try:
        while queue or live:
            while queue and len(live) < 2 * args.concurrent:
                pair = queue.popleft()
                mine, theirs = [], []
                for new_seat in (0, 1):
                    gid = 2 * pair + new_seat
                    seed = args.seed + 10_007 * pair + new_seat
                    common = {"id": gid, "draft": drafts[pair], "seed": seed,
                              "depth": args.depth, "iters": args.iters}
                    mine.append({**common, "seat": new_seat})
                    theirs.append({**common, "seat": 1 - new_seat})
                    live.append(gid)
                left, right = new.call("new", mine), old.call("new", theirs)
                for item in mine:
                    gid = item["id"]
                    assert_same(left[gid]["state"], right[gid]["state"], gid)
                    states[gid] = left[gid]["state"]

            advance(new, old, live, states)

            done = [g for g in live if states[g]["terminal"]]
            if not done:
                continue
            live = [g for g in live if not states[g]["terminal"]]
            drop = [{"id": g} for g in done]
            new.call("drop", drop)
            old.call("drop", drop)
            for gid in done:
                pair, new_seat = divmod(gid, 2)
                finished.setdefault(pair, {})[new_seat] = outcome(
                    states.pop(gid), new_seat)
            while len(pairs) in finished and len(finished[len(pairs)]) == 2:
                got = finished.pop(len(pairs))
                points.extend([got[0], got[1]])
                pairs.append(got[0] + got[1])
            write_record(out, result_record(args, points, pairs, False))
            summary = summarize(points, pairs)
            rate = len(points) / max(time.time() - started, 1e-9)
            print(f"[cross] {len(points):4d}/{args.games} "
                  f"W{summary['wins']} L{summary['losses']} D{summary['draws']} "
                  f"score {summary['score']:.3f} "
                  f"live {len(live)} {60 * rate:.1f} games/min", flush=True)
    finally:
        new.close()
        old.close()
    result = result_record(args, points, pairs, True)
    write_record(out, result)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--worker", nargs=3, metavar=("CHECKPOINT", "TRAIN_ROOT", "DEVICE"))
    ap.add_argument("--new")
    ap.add_argument("--old")
    ap.add_argument("--new-train", default="train")
    ap.add_argument("--old-train", default="/workspace/warchest-odd-stable/train")
    ap.add_argument("--old-python", default="/tmp/warchest-odd-stable-python")
    ap.add_argument("--new-device", type=int, default=0)
    ap.add_argument("--old-device", type=int, default=1)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--concurrent", type=int, default=64,
                    help="colour-swapped pairs in flight at once")
    ap.add_argument("--seed", type=int, default=83)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--out", default="runs/cross_architecture_ladder.json")
    args = ap.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        if not args.new or not args.old:
            ap.error("--new and --old are required")
        controller(args)


if __name__ == "__main__":
    main()
