#!/usr/bin/env python3
"""Play a trained ReBeL agent in the browser.

    ./play.sh [BOT] [--port 8765] [--seed N]

`BOT` is a bot directory — the same thing the arena ladders (`docs/ARENA.md`),
and it plays here exactly as it plays there: the same binary, the same search,
the same beliefs. This process is the referee, and you are the other seat.
It defaults to the most recently packed bot.

The draft is the fixed starter matchup (STARTER_WHITE / STARTER_BLACK in
engine/src/selfplay.rs, the rulebook's recommended armies); you play white, the
agent plays black. Round-start draws and the agent's moves happen
automatically; every decision that is yours appears in the UI as a
legal-action button.

The agent's private information (hand, bag composition, face-down discards)
is withheld from the snapshot the browser sees; only its public counts are
shown. Your own hand is visible because it is yours.

Only stdlib is used beyond the built `warchest` extension.
"""

import argparse
import json
import queue
import random
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEBUI = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "tools"))

import warchest  # noqa: E402
from arena import Bot  # noqa: E402

# Fixed starter matchup, mirroring engine/src/selfplay.rs.
STARTER_WHITE = [17, 12, 4, 9]  # Swordsman, Pikeman, Crossbowman, Light Cavalry
STARTER_BLACK = [1, 3, 8, 16]  # Archer, Cavalry, Lancer, Scout
# 10 control locations in board.rs order: 0,1 white starts; 2,3 black starts;
# 4..9 neutral.
LOCATION_COORDS = [(4, 0), (6, 1), (0, 5), (2, 6),
                   (2, 1), (3, 2), (5, 3), (1, 3), (3, 4), (4, 5)]


class Session:
    """One game, its agent, and the browser-facing view of it."""

    #: The human always plays white, the bot black.
    HUMAN, AGENT = 0, 1

    def __init__(self, bot: Path, seed: int | None):
        self.spec = json.loads((bot / "bot.json").read_text())
        self.name = self.spec.get("name", bot.name)
        self.search = self.spec.get("search", {})
        self.seed = seed
        self.units = {uid: (name, coins) for uid, name, coins in warchest.units_info()}
        self.geometry = self._geometry()
        self.replies = queue.Queue()
        # The bot is the same process the arena runs, spoken to over the same
        # protocol. Nothing here knows how it thinks.
        self.agent = Bot(bot, self.AGENT, -1, self.replies)
        self.log = []
        self.new_game()

    def _geometry(self):
        """Board geometry for the UI, cross-checked against the engine.

        The engine exports hex coords but not the location list, so this
        rebuilds both from the same formulas board.rs uses and verifies the
        hex indexing against the engine's own coordinate strings at startup.
        """
        coords = [tuple(c) for c in warchest.hex_coords()]
        index = {c: i for i, c in enumerate(coords)}
        locs = {i: index[c] for i, c in enumerate(LOCATION_COORDS)}
        hex_of = {f"{x},{y}": i for i, (x, y) in enumerate(coords)}
        return {"coords": coords, "locations": locs, "hex_of": hex_of}

    def check_geometry(self):
        """Verify our hex indexing against the engine's coord strings."""
        snap = self.snapshot()
        for occ in snap["board"]:
            assert self.geometry["hex_of"][occ["coord"]] == occ["hex"], occ
        for coord in snap["markers"]:
            assert coord in self.geometry["hex_of"], coord

    def new_game(self):
        seed = self.seed if self.seed is not None else random.getrandbits(63)
        self.game_id = random.getrandbits(31)
        self.log = []
        self.table = warchest.Table()
        self.table.start(self.game_id, STARTER_WHITE, STARTER_BLACK, 0,
                         [self.HUMAN, self.AGENT], seed)
        self.advance()

    def advance(self):
        """Resolve draws and the bot's replies until it is your move again."""
        while True:
            self.table.settle()
            if self.table.reap():
                return
            request = self.table.request(self.AGENT)
            if request:
                self.agent.send(request)
                _, line = self.replies.get()
                if isinstance(line, SystemExit):
                    raise line
                self.table.reply(self.AGENT, line)
                continue
            # Nothing left for the bot: whatever remains is yours. Taking the
            # human's request is what hands us the observations to narrate.
            request = self.table.request(self.HUMAN)
            if request:
                self.narrate(json.loads(request))
            return

    def narrate(self, request):
        """Turn this seat's observations into the browser's event log.

        The log is built from what the referee tells *your* seat and nothing
        else, so it cannot show you what the game does not: the agent's draws
        are counted but not named, and a coin it spends face down stays a coin
        it spent face down."""
        for ask in request.get("go", []) + request.get("watch", []):
            for obs in ask.get("obs", []):
                if obs["kind"] == "draw":
                    if obs["player"] != self.HUMAN:
                        self.log.append("Agent draws (hidden)")
                    else:
                        self.log.append(f"You draw {self.label(obs['code'])}")
                else:
                    seen = warchest.obs_label(obs["key"])
                    self.log.append(f"Agent: {seen[0].lower()}{seen[1:]}")

    @staticmethod
    def label(code):
        """Your own action or draw, named in full: it is yours to see."""
        return warchest.action_label(code).replace("Draw ", "").lower()

    def human_move(self, code):
        """Your move goes back to the referee the way a bot's does — this seat
        is not a special case, it just happens to think in a browser."""
        self.table.reply(self.HUMAN, json.dumps(
            {"done": [{"id": self.game_id, "action": code}]}))
        # Your own move never comes back as an observation — the referee only
        # tells the other seat about it — so the log records it here.
        self.log.append(f"You: {self.label(code)}")
        self.advance()

    def snapshot(self):
        snap = self.table.view(self.game_id)
        snap["human"], snap["agent"] = self.HUMAN, self.AGENT
        snap["log"] = self.log
        return sanitize(snap)


def sanitize(snap: dict) -> dict:
    """The browser's view: the agent's private zones become public counts.

    Hand, bag composition and face-down discard identities are private to the
    agent. Sizes are public (you can see how many coins someone holds).
    """
    agent = snap["agent"]
    z = snap["zones"][agent]
    z["hand_size"] = sum(z["hand"].values())
    z["bag_size"] = sum(z["bag"].values())
    z["facedown_count"] = sum(z["facedown_discard"].values())
    for key in ("hand", "bag", "facedown_discard"):
        z[key] = {"hidden": True}
    return snap


LOCK = threading.Lock()
SESSION: Session | None = None


def send_json(handler: BaseHTTPRequestHandler, obj, status=200):
    body = json.dumps(obj).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: BaseHTTPRequestHandler):
    n = int(handler.headers.get("Content-Length", 0))
    return json.loads(handler.rfile.read(n)) if n else {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = (WEBUI / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/state":
            with LOCK:
                snap = SESSION.snapshot()
            send_json(self, snap)
        elif path == "/api/meta":
            with LOCK:
                s = SESSION
            send_json(self, {
                "units": {str(uid): name for uid, (name, _) in s.units.items()},
                "geometry": s.geometry,
                "draft": {
                    "white": [{"id": u, "name": s.units[u][0]} for u in STARTER_WHITE],
                    "black": [{"id": u, "name": s.units[u][0]} for u in STARTER_BLACK],
                },
                "human_seat": 0,
                "agent_seat": 1,
                "bot": s.name,
                "search": s.search,
            })
        else:
            send_json(self, {"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/move":
            req = read_json(self)
            with LOCK:
                try:
                    SESSION.human_move(int(req["code"]))
                    snap = SESSION.snapshot()
                except Exception as e:  # illegal action / not your turn
                    send_json(self, {"error": str(e)}, 400)
                    return
                send_json(self, snap)
        elif path == "/api/new":
            with LOCK:
                SESSION.new_game()
                send_json(self, SESSION.snapshot())
        else:
            send_json(self, {"error": "not found"}, 404)


def latest_bot() -> Path:
    """The most recently packed bot that carries weights."""
    bots = sorted((p.parent for p in (ROOT / "bots").glob("*/weights.bin")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not bots:
        raise SystemExit(
            "no bots found. Pack one:\n"
            "  python tools/arena.py pack runs/<run> --snapshot final --name mine")
    return bots[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bot", type=Path, nargs="?", default=None,
                    help="bot directory (default: the newest under bots/)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seed", type=int, default=None, help="fixed RNG seed for draws")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    global SESSION
    bot = args.bot or latest_bot()
    SESSION = Session(bot, args.seed)
    SESSION.check_geometry()
    print(f"bot: {SESSION.name} ({bot})")
    print(f"you: {', '.join(SESSION.units[u][0] for u in STARTER_WHITE)}")
    print(f"agent: {', '.join(SESSION.units[u][0] for u in STARTER_BLACK)} (black)")
    print(f"agent search: {SESSION.search}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"playing at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
