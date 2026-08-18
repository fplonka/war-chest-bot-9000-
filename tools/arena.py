#!/usr/bin/env python3
"""Play archived bots against each other.

    tools/arena.py ladder bots/v5-2h bots/v4-12h bots/greedy --games 200
    tools/arena.py probe  bots/v5-2h --probe bots/lbr --games 200
    tools/arena.py pack   runs/v5_d2_125m --snapshot final --name v5-2h

A bot is a directory holding a binary, its weights and a `bot.json`. The
binary is built once, at the revision that trained the weights, and never
rebuilt — which is the only way to keep playing an architecture after the
source that produced it has been rewritten.

The referee owns the true position and the dice and speaks to each bot over a
pipe, in public information only. Neither side waits for the other's slowest
game: a bot answers each game as that game is ready, so it always holds a
mixture of games and both its cores and its device stay busy. `docs/ARENA.md`
has the protocol.

Ratings are Bradley-Terry, quoted against the first bot on the command line.
Elo does not carry between separate ladders, so a comparison worth trusting is
one where both bots played in the same run.
"""

import argparse
import collections
import itertools
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

import warchest  # noqa: E402

# Every unit in scope for the ranked two-player game. A ladder drafts from all
# of it, so a rating is strength across the pool rather than on one army.
DRAFT_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 52, 53, 54]

PROTOCOL = 5

# A game is won by controlling this many locations.
WIN_MARKERS = 6


# ------------------------------------------------------------------ bot files

class Bot:
    """One bot process, and the manifest that describes it."""

    def __init__(self, path, seat, device, replies):
        self.dir = Path(path)
        self.seat = seat
        self.closing = False
        # Both bots answer into one queue, so the referee waits until *either*
        # has something rather than on a particular one.
        self.replies = replies
        self.spec = json.loads((self.dir / "bot.json").read_text())
        self.name = self.spec.get("name", self.dir.name)
        argv = [str(self.dir / "bot"), "--name", self.name,
                "--mind", self.spec.get("mind", "rebel"),
                "--device", str(device)]
        if self.spec.get("weights"):
            argv += ["--weights", str(self.dir / self.spec["weights"])]
        for key, flag in (("depth", "--depth"), ("iters", "--iters"),
                          ("cfr", "--cfr"), ("temp", "--temp"),
                          ("node_cap", "--node-cap")):
            if key in self.spec.get("search", {}):
                argv += [flag, str(self.spec["search"][key])]
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            bufsize=1)
        hello = json.loads(self._line())
        if hello["protocol"] != PROTOCOL:
            raise SystemExit(
                f"{self.name} speaks protocol {hello['protocol']}, not {PROTOCOL}")
        if hello["rules"] != warchest.rules_table_hash():
            raise SystemExit(
                f"{self.name} was built against different rules; no result "
                f"between it and this referee would mean anything")
        threading.Thread(target=self._read, daemon=True).start()

    def _line(self):
        """One protocol line. A bot's stdout carries nothing else, so anything
        that is not a message is a bug worth naming rather than a parse error
        three frames up."""
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"{self.name} exited with {self.proc.wait()}")
        if not line.startswith("{"):
            raise SystemExit(f"{self.name} wrote this to stdout: {line.strip()!r}")
        return line

    def _read(self):
        """Replies arrive as games finish rather than in step with the
        requests, so they are drained on their own thread. A bot that stops on
        its own is an error the referee must see; one this process shut down is
        not."""
        try:
            while True:
                self.replies.put((self.seat, self._line()))
        except SystemExit as stop:
            if not self.closing:
                self.replies.put((self.seat, stop))

    def send(self, request):
        self.proc.stdin.write(request + "\n")
        self.proc.stdin.flush()

    def close(self):
        self.closing = True
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)


# ---------------------------------------------------------------- the matches

def drafts(seed, count):
    """`count` paired drafts. Drawn from one sequential stream so the same seed
    reproduces the same ladder however the pairs are later shared out."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        units = rng.sample(DRAFT_POOL, 8)
        out.append({"white": units[:4], "black": units[4:],
                    "first": rng.randrange(2)})
    return out


def apply_reply(table, reply):
    seat, line = reply
    if isinstance(line, SystemExit):
        raise line
    table.reply(seat, line)


def match(bots, pairs, replies, seed, concurrent, report, probe=None):
    """Play every draft twice with the colours swapped, and return A's score in
    each game. Colour-swapped pairs over a shared draft remove most of the
    variance that comes from the draft itself.

    Nothing here waits for a batch. Both bots are handed every game they are
    not already thinking about, and whichever game either of them finishes
    first is carried forward at once. That is what keeps each bot holding a
    mixture of games — some being built on its cores, some being solved on its
    device — instead of swinging between the two."""
    table = warchest.Table()
    if probe is not None:
        table.probe(probe)
    todo = deque(range(2 * len(pairs)))
    scores = {}
    started = time.time()
    while todo or table.live() or table.outstanding():
        while todo and table.live() < concurrent:
            gid = todo.popleft()
            draft, swapped = pairs[gid // 2], gid % 2
            # `bots[seat]`: A takes black in the second game of each pair.
            table.start(gid, draft["white"], draft["black"], draft["first"],
                        [1, 0] if swapped else [0, 1], seed + 7919 * gid)
        table.settle()
        for gid, white, _, z in table.reap():
            scores[gid] = z if white == 0 else -z
            report(len(scores), time.time() - started)
        for index, bot in enumerate(bots):
            request = table.request(index)
            if request:
                bot.send(request)
        if not table.outstanding():
            continue
        # Wait for the first reply, then take everything that has piled up
        # behind it before going round again.
        apply_reply(table, replies.get())
        while True:
            try:
                apply_reply(table, replies.get_nowait())
            except queue.Empty:
                break
    return [scores[gid] for gid in sorted(scores)]


# ------------------------------------------------------------------- ratings

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
    """The standard error of each rating, from the Fisher information of the
    games actually played. A pair that went 20-0 carries almost none."""
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


def sign_test(wins, losses):
    """Two-sided sign test. `n` decisive trials, `max(w, l)` or better by
    chance alone."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(a, b, points):
    """One pairing as it gets read.

    Games come in colour-swapped pairs over a shared draft, so the pair — not
    the game — is the independent trial, and the paired test is the one that
    answers "is A actually better". The per-game record is there for the Elo
    fit and for reading the margin."""
    w = sum(p > 0 for p in points)
    d = sum(p == 0 for p in points)
    pairs = [points[i] + points[i + 1] for i in range(0, len(points) - 1, 2)]
    won = sum(p > 0 for p in pairs)
    lost = sum(p < 0 for p in pairs)
    return {"a": a, "b": b, "w": w, "l": len(points) - w - d, "d": d,
            "n": len(points), "score": round((w + 0.5 * d) / len(points), 4),
            "paired_w": won, "paired_l": lost,
            "paired_p": float(f"{sign_test(won, lost):.4g}")}


def ratings(names, records):
    k = len(names)
    n = np.zeros((k, k))
    score = np.zeros((k, k))
    for (i, j), points in records.items():
        n[i][j] = n[j][i] = len(points)
        score[i][j] = sum(1.0 if p > 0 else 0.5 if p == 0 else 0.0 for p in points)
        score[j][i] = len(points) - score[i][j]
    # One phantom drawn game per played pair keeps a clean sweep finite.
    played = (n > 0) * (1 - np.eye(k))
    elo = fit_elo(n + played, score + 0.5 * played)
    se = elo_stderr(n, elo)
    # A bot that has not played yet has no rating, and one that has never been
    # held to a close game has no error bar. JSON has no infinity, so both are
    # null rather than something a reader has to guess at.
    finite = lambda v: round(float(v), 1) if math.isfinite(v) else None
    return [{"name": names[i],
             "elo": finite(elo[i]) if n[i].sum() else None,
             "se": finite(se[i]),
             "games": int(n[i].sum()),
             "score": round(float(score[i].sum() / max(n[i].sum(), 1)), 3)}
            for i in range(k)]


# ----------------------------------------------------------------- the probe

def probe(bot_path, probe_path, games, seed, concurrent, devices, out_path):
    """What knowing the opponent's actual strategy is worth.

    The same probe plays the same bot over the same games twice. The second
    time the referee shows it the strategy behind every move; nothing else
    differs, so the change in score is what that knowledge bought.

    A bot models its opponent as a copy of itself, and this replaces that model
    with the truth — so the number is the cost of the modelling assumption. It
    is a lower bound on exploitability, not a measure of it: a bot probed by a
    copy of itself scores zero, which is a check on the belief filter rather
    than a verdict on the bot. Aim it at something the probe does not
    resemble."""
    replies = queue.Queue()
    spec = json.loads((Path(bot_path) / "bot.json").read_text())
    name = spec.get("name", Path(bot_path).name)
    pairs = drafts(seed, games // 2)
    subject = Bot(bot_path, 0, devices[0], replies)
    prober = Bot(probe_path, 1, devices[1], replies)
    rows = {}
    try:
        for peeking in (False, True):
            points = match((subject, prober), pairs, replies, seed, concurrent,
                           lambda *a: None, probe=1 if peeking else None)
            key = "peeking" if peeking else "blind"
            # `match` scores the subject; the probe's score is the other side.
            rows[key] = summarize("probe", name, [-p for p in points])
            print(f"  probe {'sees' if peeking else 'does not see'} the "
                  f"strategy: scores {rows[key]['score']:.3f}", flush=True)
    finally:
        subject.close()
        prober.close()

    gain = rows["peeking"]["score"] - rows["blind"]["score"]
    result = {"kind": "probe", "bot": name, "probe": prober.name,
              "games": games, "seed": seed,
              "blind": rows["blind"], "peeking": rows["peeking"],
              "gain": round(gain, 4)}
    write_json(out_path, result)
    print(f"\n{name} scores {-rows['blind']['score']:+.3f} against "
          f"{prober.name} playing blind, {-rows['peeking']['score']:+.3f} "
          f"once {prober.name} is shown its strategy.")
    print(f"{name}: that knowledge is worth {gain:+.3f} of a game to the "
          f"probe (a lower bound on exploitability; zero is what a copy of "
          f"itself should score).")
    print(f"wrote {out_path}")
    return result


# ------------------------------------------------------------- the tablebase

def generate(paths, games, seed, concurrent, devices, out_dir,
             depth=8, per_bucket=400, min_plies=2, budget=25_000):
    """Find positions whose result is proven, by watching bots play.

    Positions come from games between *bots*, which is what stops this rotting:
    a bot is a frozen binary behind a protocol, so nothing here knows or cares
    what architecture produced the games. Put `bots/random` on both sides and
    the source is random play; put a trained pair there and the positions are
    ones a real game reaches. Same code either way, and an architecture change
    cannot break it.

    A position is kept when the side to move wins whatever the other does —
    within `depth` plies, and against every hand the opponent could be holding.
    The quota is per hardness band — what share of the legal moves keep the win,
    crossed with whether any hidden information is left — because without one
    the set fills with positions where most moves win, and those are far more
    common than the sharp ones and tell you much less.

    `budget` caps the nodes one proof may build. It never lets an unproven
    position through — it only decides which positions are cheap enough to
    settle — and the few it gives up on cost far more than the rest put
    together. Raising it to 400,000 finds 1.5% more positions and takes 2.6
    times as long, so the time is better spent on more games."""
    replies = queue.Queue()
    bots = tuple(Bot(p, i, devices[i % len(devices)], replies)
                 for i, p in enumerate(paths))
    table = warchest.Table()
    pairs = drafts(seed, (games + 1) // 2)
    todo = deque(range(games))
    # Enough positions to keep every core busy for a sweep; the proof is the
    # only expensive thing here and it parallelises across positions.
    batch = 8 * (os.cpu_count() or 8)
    taken, questions, seen = {}, [], set()
    proven = 0
    published = 0
    started = time.time()
    out_dir = Path(out_dir)

    def publish():
        """Write what has been found so far. A long run that only lands at the
        end is one crash away from having been a waste of the night."""
        write_json(out_dir / "suite.json", {
            "kind": "tablebase", "games": games, "depth": depth, "seed": seed,
            "min_plies": min_plies, "per_bucket": per_bucket,
            "budget": budget,
            "players": [b.name for b in bots],
            "proven": proven, "questions": questions,
        })

    def harvest(batch):
        """Take every proven position the referee found, subject to quota.

        The proving is the whole cost of this program and it happens inside
        the referee, over all live games at once and off the interpreter
        lock. What is left here is bookkeeping."""
        nonlocal proven
        for _, winner, plies, wins, moves, size, position in table.harvest(
                batch, min_plies, depth, WIN_MARKERS - 2, budget):
            proven += 1
            key = (band(wins, moves), size > 1)
            if position in seen or taken.get(key, 0) >= per_bucket:
                continue
            seen.add(position)
            taken[key] = taken.get(key, 0) + 1
            questions.append({
                "position": position, "winner": winner, "depth": plies,
                "wins": wins, "moves": moves, "range": size,
            })

    try:
        while todo or table.live():
            while todo and table.live() < concurrent:
                gid = todo.popleft()
                draft, swapped = pairs[gid // 2], gid % 2
                table.start(gid, draft["white"], draft["black"], draft["first"],
                            [1, 0] if swapped else [0, 1], seed + 7919 * gid)
            table.settle()
            table.reap()
            harvest(batch)
            requests = [table.request(i) for i in range(len(bots))]
            for bot, request in zip(bots, requests):
                if request:
                    bot.send(request)
            if not table.outstanding():
                continue
            apply_reply(table, replies.get())
            while True:
                try:
                    apply_reply(table, replies.get_nowait())
                except queue.Empty:
                    break
            if len(questions) >= published + 100:
                published = len(questions)
                publish()
                rate = len(questions) / max(time.time() - started, 1e-9)
                print(f"  {len(questions)} kept of {proven} proven, "
                      f"{60 * rate:.0f}/min", flush=True)
        harvest(0)
    finally:
        for bot in bots:
            bot.close()

    publish()
    print(f"\n{len(questions)} questions from {games} games between "
          f"{' and '.join(b.name for b in bots)}")
    print(f"{'moves that win':>14} {'known':>7} {'hidden':>7}")
    for name, _, _ in SHARES:
        k, h = taken.get((name, False), 0), taken.get((name, True), 0)
        if k or h:
            print(f"{name:>14} {k:>7} {h:>7}")
    plies = collections.Counter(q["depth"] for q in questions)
    print(f"{'plies to win':>14} {'count':>7}")
    for d in sorted(plies):
        print(f"{d:>14} {plies[d]:>7}")
    print(f"wrote {out_dir / 'suite.json'}")


def tablebase(bot_path, suite_dir, out_path, device=-1, concurrent=32):
    """Ask a bot the questions whose answers are proven.

    Each question is a position in which one side wins whatever the other
    does. The bot is seated there — the position carries both ranges, because
    the proof was carried out against those ranges and a question has to mean
    the same thing to every bot — plays the winning side, and makes one move.
    The move is right exactly when the win is still forced afterwards, which
    the rules decide, not a network and not an opponent.

    So this is the one measurement here that is true rather than relative: the
    same questions and the same marking for any bot ever built.

    Questions are independent, so many are in flight at once; a searching bot
    that answered them one at a time would leave its card idle between each."""
    meta = json.loads((Path(suite_dir) / "suite.json").read_text())
    questions = meta["questions"]
    spec = json.loads((Path(bot_path) / "bot.json").read_text())
    name = spec.get("name", Path(bot_path).name)

    replies = queue.Queue()
    bot = Bot(bot_path, 0, device, replies)
    table = warchest.Table()
    kept, blundered = [], []

    def ask(i):
        q = questions[i]
        # Our bot takes the winning seat; nothing plays the other one.
        table.start_at(i, q["position"], [0, 1] if q["winner"] == 0 else [1, 0], 1)
        return {"id": i, "at": {"position": q["position"], "seat": q["winner"],
                                "seed": 1}}

    try:
        sent = min(concurrent, len(questions))
        bot.send(json.dumps({"go": [ask(i) for i in range(sent)]}))
        while len(kept) + len(blundered) < len(questions):
            _, line = replies.get()
            if isinstance(line, SystemExit):
                raise line
            reply = json.loads(line)
            if reply.get("error"):
                raise SystemExit(f"{name} on question {reply.get('id')}: "
                                 f"{reply['error']}")
            table.reply(0, line)
            answered = [d["id"] for d in reply["done"]]
            for i in answered:
                # One ply has gone, so a correct move leaves the win forced in
                # one ply fewer. The marking runs at the default budget, which
                # is far above the one generation used: a move must never be
                # called a blunder because the prover ran out of nodes, and
                # erring high can only ever credit a move that really wins.
                q = questions[i]
                (kept if table.forced(i, q["winner"], max(q["depth"] - 1, 0))
                 else blundered).append(i)
            more = range(sent, min(sent + len(answered), len(questions)))
            sent += len(more)
            bot.send(json.dumps({"drop": answered,
                                 "go": [ask(i) for i in more]}))
            done = len(kept) + len(blundered)
            if done % 100 == 0:
                print(f"  {done}/{len(questions)}  held {len(kept)}", flush=True)
    finally:
        bot.close()

    n = len(kept) + len(blundered)
    held = set(kept)

    # A single average hides everything that matters here, because the set
    # deliberately spans questions of very different hardness.
    def slice_of(pick):
        idx = [i for i, q in enumerate(questions) if pick(q)]
        got = sum(1 for i in idx if i in held)
        return {"n": len(idx), "held": got,
                "rate": round(got / max(len(idx), 1), 4)}

    depths = sorted({q["depth"] for q in questions})
    result = {
        "kind": "tablebase", "bot": name, "suite": str(suite_dir),
        "questions": n, "held": len(kept),
        "rate": round(len(kept) / max(n, 1), 4),
        "by_depth": {str(d): slice_of(lambda q, d=d: q["depth"] == d) for d in depths},
        # What share of the legal moves kept the win. This is the hardness
        # axis: three right moves out of four is an easy position and three
        # out of thirty is not, so the count alone says little.
        "by_share": {name: slice_of(lambda q, lo=lo, hi=hi:
                                    lo < q["wins"] / q["moves"] <= hi)
                     for name, lo, hi in SHARES},
        "hidden": slice_of(lambda q: q["range"] > 1),
        "known": slice_of(lambda q: q["range"] == 1),
    }
    write_json(out_path, result)
    print(f"\n{name}: kept the win in {len(kept)}/{n} proven positions "
          f"({100 * result['rate']:.1f}%)")
    buckets("moves that win", [(n, result["by_share"][n]) for n, _, _ in SHARES])
    buckets("plies to win", [(d, result["by_depth"][str(d)]) for d in depths])
    buckets("hidden hands", [("none", result["known"]), ("some", result["hidden"])])
    print(f"wrote {out_path}")
    return result


# ---------------------------------------------------------------- the ladder

def schedule(count):
    """Every pairing, ordered so that consecutive ones keep a bot on the seat it
    already holds. Starting a bot costs a weights load and a kernel build, so a
    ladder that reseats one side at a time pays that far less often."""
    pairs = list(itertools.combinations(range(count), 2))
    out, seated = [], (None, None)
    while pairs:
        best = max(pairs, key=lambda p: sum(x in seated for x in p))
        pairs.remove(best)
        if best[1] == seated[0] or best[0] == seated[1]:
            best = (best[1], best[0])
        seated = best
        out.append(best)
    return out


def report(names, paths, specs, records, games, seed, complete):
    players = ratings(names, records)
    for player, spec in zip(players, specs):
        if spec.get("minutes") is not None:
            player["minutes"] = spec["minutes"]
    return {
        "kind": "ladder", "complete": complete,
        "bots": [{"name": n, "path": str(p),
                  **{k: v for k, v in s.items() if k != "name"}}
                 for n, p, s in zip(names, paths, specs)],
        "games": games, "seed": seed, "anchor": names[0],
        "players": players,
        "pairs": [summarize(names[i], names[j], points)
                  for (i, j), points in records.items()],
    }


def name(bot_dir):
    return Path(bot_dir).name


def devices(spec):
    return [int(d) for d in spec.split(",")]


#: Hardness bands, by the share of legal moves that keep the win. A position
#: where one move in ten is right is a different question from one where half
#: of them are, however deep the win is.
SHARES = [("under 10%", 0.0, 0.10), ("10-25%", 0.10, 0.25),
          ("25-50%", 0.25, 0.50), ("over 50%", 0.50, 1.0)]


def band(wins, moves):
    """Which hardness band a position falls in."""
    share = wins / moves
    return next(n for n, lo, hi in SHARES if lo < share <= hi)


def buckets(title, rows):
    """One rate per bucket, with the count it rests on. A rate over nine
    questions says less than the same rate over ninety."""
    print(f"{title:>14} {'n':>6} {'held':>6} {'rate':>7}")
    for label, row in rows:
        if row["n"]:
            print(f"{label:>14} {row['n']:>6} {row['held']:>6} {row['rate']:>7.3f}")


def report_path(given, stem):
    """Where a report lands. Reports go to `arena/` so the dashboard finds
    them without being told."""
    return Path(given) if given else Path("arena") / f"{stem}.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # `allow_nan` off: Python would happily write `Infinity`, which no other
    # reader accepts. Better to fail here than to ship a file nothing can parse.
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)


def ladder(paths, games, seed, concurrent, devices, out_path):
    if games < 2 or games % 2:
        raise SystemExit("--games must be a positive even number")
    specs = [json.loads((Path(p) / "bot.json").read_text()) for p in paths]
    names = [s.get("name", Path(p).name) for s, p in zip(specs, paths)]
    if len(set(names)) != len(names):
        raise SystemExit(f"bot names must be distinct: {names}")

    records = {}
    running = {}
    replies = queue.Queue()

    def bot_on(seat, index):
        """The bot seated at `seat`, started on that seat's device if it is not
        already the one sitting there."""
        if running.get(seat) and running[seat][0] == index:
            return running[seat][1]
        if running.get(seat):
            running[seat][1].close()
        running[seat] = (index, Bot(paths[index], seat, devices[seat], replies))
        return running[seat][1]

    try:
        for i, j in schedule(len(paths)):
            bots = (bot_on(0, i), bot_on(1, j))
            # Each pairing gets its own drafts and its own draw stream, so two
            # pairings are independent evidence rather than the same games
            # played by different bots.
            pairing_seed = seed + 1009 * i + 101 * j

            def progress(done, secs, i=i, j=j):
                if done % 20 == 0 or done == games:
                    print(f"  {names[i]} vs {names[j]}  {done}/{games}  "
                          f"{60 * done / max(secs, 1e-9):.0f} games/min",
                          flush=True)

            records[(i, j)] = match(bots, drafts(pairing_seed, games // 2),
                                    replies, pairing_seed, concurrent, progress)
            got = summarize(names[i], names[j], records[(i, j)])
            print(f"{names[i]:>20s} vs {names[j]:<20s} "
                  f"W{got['w']:3d} L{got['l']:3d} D{got['d']:3d}  "
                  f"score {got['score']:.3f}  "
                  f"paired {got['paired_w']}-{got['paired_l']} "
                  f"p={got['paired_p']:.2g}", flush=True)
            # Published after every pairing: a ten-pairing ladder takes half an
            # hour, and the dashboard should show it filling in.
            write_json(out_path,
                       report(names, paths, specs, records, games, seed, False))
    finally:
        for _, bot in running.values():
            bot.close()

    result = report(names, paths, specs, records, games, seed, complete=True)
    write_json(out_path, result)

    print(f"\n=== Elo ({names[0]} = 0) ===")
    print(f"{'bot':>24s} {'elo':>7s} {'95%':>6s} {'games':>6s} {'score':>7s}")
    for p in sorted(result["players"], key=lambda p: -(p["elo"] or -1e9)):
        elo = "—" if p["elo"] is None else f"{p['elo']:.0f}"
        ci = "—" if p["se"] is None else f"{1.96 * p['se']:.0f}"
        print(f"{p['name']:>24s} {elo:>7s} {ci:>6s} "
              f"{p['games']:>6d} {p['score']:>7.3f}")
    print(f"\nwrote {out_path}")
    return result


# -------------------------------------------------------------- run snapshots

def pack(run, binary, out_dir, snapshot=None, name=None):
    """Archive training snapshots as bots.

    A bot directory is self-contained: the binary that can play these weights,
    the weights, and a manifest. Nothing outside it is ever needed again, which
    is what lets an architecture keep playing after the source that produced it
    has been rewritten.

    Packing a whole run is how a run's own progress gets rated — the same
    operation, and the same ladder, as rating one architecture against
    another.
    """
    import shutil

    import torch
    from export_weights import write_bin, load

    if not Path(binary).exists():
        raise SystemExit(f"{binary} does not exist. Build it:\n"
                         f"  cd engine && cargo build --release --bin bot")

    run = Path(run)
    log = json.loads((run / "log.json").read_text())
    cfg = log.get("cfg", {})
    snaps = [s for s in log.get("snapshots", [])
             if snapshot is None or s["label"] == snapshot]
    if not snaps:
        raise SystemExit(f"{run} has no snapshot {snapshot!r}")
    if name and len(snaps) > 1:
        raise SystemExit("--name needs a single --snapshot")
    made = []
    for snap in snaps:
        checkpoint = torch.load(run / snap["file"], map_location="cpu",
                                weights_only=False)
        search = {"depth": cfg.get("depth", 2), "iters": cfg.get("iters", 64),
                  "cfr": cfg.get("cfr", "dcfr")}
        search.update(checkpoint.get("search") or {})
        bot = name or f"{run.name}.{snap['label']}"
        directory = out_dir / bot
        directory.mkdir(parents=True, exist_ok=True)
        write_bin(load(run / snap["file"]), directory / "weights.bin")
        shutil.copy2(binary, directory / "bot")
        (directory / "bot.json").write_text(json.dumps({
            "name": bot,
            "sha": checkpoint.get("git", ""),
            "mind": "rebel",
            "weights": "weights.bin",
            "search": {k: v for k, v in search.items()
                       if k in ("depth", "iters", "cfr")},
            "minutes": round(snap["t"] / 60.0, 1),
            "note": f"{run.name} {snap['label']}, {snap['t'] / 60:.0f} min",
        }, indent=1) + "\n")
        made.append(directory)
        print(f"packed {directory}")
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    lad = sub.add_parser("ladder", help="round robin over bot directories")
    lad.add_argument("bots", nargs="+")
    lad.add_argument("--games", type=int, default=200,
                     help="games per pairing, split evenly between colours")
    lad.add_argument("--seed", type=int, default=83)
    lad.add_argument("--concurrent", type=int, default=128,
                     help="games in flight; this is what fills the device")
    lad.add_argument("--devices", default="0,1",
                     help="the two sides' CUDA devices; use 0,0 for one card")
    lad.add_argument("--out", default="")

    pr = sub.add_parser(
        "probe", help="what knowing a bot's actual strategy is worth")
    pr.add_argument("bot")
    pr.add_argument("--probe", default="bots/lbr",
                    help="the probing bot directory")
    pr.add_argument("--games", type=int, default=200)
    pr.add_argument("--seed", type=int, default=83)
    pr.add_argument("--concurrent", type=int, default=96)
    pr.add_argument("--devices", default="0,1")
    pr.add_argument("--out", default="")

    gen = sub.add_parser("generate",
                         help="find positions with proven answers, by watching bots play")
    gen.add_argument("bots", nargs=2,
                     help="the two bots whose games the positions come from")
    gen.add_argument("--games", type=int, default=20000)
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--depth", type=int, default=8)
    gen.add_argument("--per-bucket", type=int, default=400)
    gen.add_argument("--budget", type=int, default=25_000)
    gen.add_argument("--min-plies", type=int, default=2,
                     help="a win available sooner is a different, easier question")
    gen.add_argument("--concurrent", type=int, default=192,
                     help="games in flight; measured best on two cards")
    gen.add_argument("--devices", default="-1,-1")
    gen.add_argument("--out", default="suites/forced")

    tb = sub.add_parser("tablebase",
                        help="score a bot on positions with proven answers")
    tb.add_argument("bot")
    tb.add_argument("--suite", default="suites/forced")
    tb.add_argument("--device", type=int, default=-1)
    tb.add_argument("--concurrent", type=int, default=32)
    tb.add_argument("--out", default="")

    pk = sub.add_parser("pack", help="archive run snapshots as bots")
    pk.add_argument("run")
    pk.add_argument("--snapshot", default=None,
                    help="one snapshot label instead of all of them")
    pk.add_argument("--name", default=None, help="bot name (default run.label)")
    pk.add_argument("--bin", default=str(ROOT / "engine/target/release/bot"),
                    help="the binary to freeze into the bot")
    pk.add_argument("--out", default="bots")

    args = ap.parse_args()
    if args.command == "pack":
        return pack(args.run, Path(args.bin).resolve(), Path(args.out),
                    args.snapshot, args.name)
    if args.command == "generate":
        return generate(args.bots, args.games, args.seed, args.concurrent,
                        devices(args.devices), args.out, args.depth,
                        args.per_bucket, args.min_plies, args.budget)
    if args.command == "tablebase":
        return tablebase(args.bot, args.suite,
                         report_path(args.out, f"tablebase_{name(args.bot)}"),
                         args.device, args.concurrent)
    if len(devices(args.devices)) != 2:
        ap.error("--devices takes exactly two device ids")
    if args.command == "probe":
        return probe(args.bot, args.probe, args.games, args.seed,
                     args.concurrent, devices(args.devices),
                     report_path(args.out, f"probe_{name(args.bot)}"))
    stem = "_vs_".join(name(b) for b in args.bots[:3])
    ladder(args.bots, args.games, args.seed, args.concurrent,
           devices(args.devices), report_path(args.out, stem))


if __name__ == "__main__":
    main()
