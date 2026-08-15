#!/usr/bin/env python3
"""Play checkpoints from incompatible engine revisions in one exact ladder."""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

DRAFT_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 52, 53, 54]


def worker(checkpoint, train_root, device):
    sys.path.insert(0, train_root)
    import warchest
    from export_weights import load

    net = load(checkpoint)
    net.push(0)
    w, b, ln = net.flat()
    warchest.gpu_start(net.dims, w, b, ln, devices=[int(device)])
    warchest.set_cap_value(0.0)
    game = None
    for line in sys.stdin:
        try:
            req = json.loads(line)
            op = req["op"]
            if op == "new":
                game = warchest.LiveGame(
                    req["draft"], req["seat"], 0, req["depth"], req["iters"],
                    req["seed"], True)
                result = {"state": game.snapshot()}
            elif op == "chance":
                result = {"state": game.relay_chance()}
            elif op == "move":
                move = game.relay_agent_move()
                result = {"move": move, "state": game.snapshot()}
            elif op == "apply":
                move = req["move"]
                action = {"code": move["code"]}
                belief = [tuple(row) for row in move["belief"]]
                result = {"state": game.relay_apply(action, belief)}
            elif op == "stop":
                break
            else:
                raise ValueError(f"unknown operation {op}")
            print("@@" + json.dumps(result, separators=(",", ":")), flush=True)
        except Exception as exc:
            print("@@" + json.dumps({"error": repr(exc)}), flush=True)


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

    def call(self, op, **args):
        self.proc.stdin.write(json.dumps({"op": op, **args}) + "\n")
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
            return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write('{\"op\":\"stop\"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)


def comparable(state):
    return {k: v for k, v in state.items()
            if k not in {"agent", "human", "log", "actions"}}


def assert_same(a, b):
    left, right = comparable(a), comparable(b)
    if left != right:
        raise RuntimeError(
            "engine states diverged:\n" + json.dumps({"new": left, "old": right}, indent=2))


def play(new, old, draft, new_seat, seed, depth, iters):
    ns = new.call("new", draft=draft, seat=new_seat, seed=seed,
                  depth=depth, iters=iters)["state"]
    os_ = old.call("new", draft=draft, seat=1 - new_seat, seed=seed,
                   depth=depth, iters=iters)["state"]
    assert_same(ns, os_)
    while not ns["terminal"]:
        if ns["is_chance"]:
            ns = new.call("chance")["state"]
            os_ = old.call("chance")["state"]
        else:
            actor, peer = (new, old) if ns["to_act"] == new_seat else (old, new)
            result = actor.call("move")
            other = peer.call("apply", move=result["move"])
            if actor is new:
                ns, os_ = result["state"], other["state"]
            else:
                os_, ns = result["state"], other["state"]
        assert_same(ns, os_)
    winner = ns["winner"]
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

    script = str(Path(__file__).resolve())
    new = Engine(
        "new", args.python, script, args.new, args.new_train, args.new_device)
    old_env = {"PYTHONPATH": args.old_python}
    old = Engine(
        "old", args.python, script, args.old, args.old_train, args.old_device, old_env)
    rng = random.Random(args.seed)
    try:
        for pair in range(args.games // 2):
            units = rng.sample(DRAFT_POOL, 8)
            draft = {
                "white_units": units[:4],
                "black_units": units[4:],
                "first_player": "white" if rng.randrange(2) == 0 else "black",
            }
            if pair < len(pairs):
                continue
            pair_points = []
            for new_seat in (0, 1):
                game_seed = args.seed + 10_007 * pair + new_seat
                point = play(new, old, draft, new_seat, game_seed, args.depth, args.iters)
                points.append(point)
                pair_points.append(point)
                summary = summarize(
                    points,
                    pairs + [sum(pair_points)] if len(pair_points) == 2 else pairs,
                )
                outcome = "W" if point == 1 else "L" if point == 0 else "D"
                print(
                    f"[cross] {len(points):4d}/{args.games} new={outcome} "
                    f"W{summary['wins']} L{summary['losses']} D{summary['draws']}",
                    flush=True,
                )
            pairs.append(sum(pair_points))
            write_record(out, result_record(args, points, pairs, False))
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
