#!/usr/bin/env python3
"""Play a trained ReBeL agent in the browser.

    ./play.sh [--ckpt PATH] [--port 8765] [--depth 2] [--iters 16] [--seed N]

The checkpoint defaults to the most recent run under `runs/` that has a
`ckpt_final.pt`. The draft is the fixed starter matchup (STARTER_WHITE /
STARTER_BLACK in engine/src/selfplay.rs, the rulebook's recommended armies);
you play white, the agent plays black. Round-start draws and the agent's
moves happen automatically; every decision that is yours appears in the UI as
a legal-action button.

The agent's private information (hand, bag composition, face-down discards)
is withheld from the snapshot the browser sees; only its public counts are
shown. Your own hand is visible because it is yours.

Only stdlib is used beyond the repo's own dependencies (torch to read the
checkpoint, the built `warchest` extension).
"""

import argparse
import json
import random
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEBUI = Path(__file__).resolve().parent

# The repo's tools load `train.py` by putting train/ itself on sys.path
# (`import train` then resolves to train/train.py).
sys.path.insert(0, str(ROOT / "train"))

import warchest  # noqa: E402
from export_weights import load as load_ckpt  # noqa: E402

# Fixed starter matchup, mirroring engine/src/selfplay.rs.
STARTER_WHITE = [17, 12, 4, 9]  # Swordsman, Pikeman, Crossbowman, Light Cavalry
STARTER_BLACK = [1, 3, 8, 16]  # Archer, Cavalry, Lancer, Scout
# 10 control locations in board.rs order: 0,1 white starts; 2,3 black starts;
# 4..9 neutral.
LOCATION_COORDS = [(4, 0), (6, 1), (0, 5), (2, 6),
                   (2, 1), (3, 2), (5, 3), (1, 3), (3, 4), (4, 5)]


class Session:
    """One game, its agent, and the browser-facing view of it."""

    def __init__(self, ckpt: Path, depth: int, iters: int, seed: int | None):
        self.ckpt = ckpt
        self.depth = depth
        self.iters = iters
        self.seed = seed
        self.units = {uid: (name, coins) for uid, name, coins in warchest.units_info()}
        self.geometry = self._geometry()
        net = load_ckpt(str(ckpt))
        net.push(0)
        # The real game's terminal payoff: the horizon marker bonus is a
        # training-time device, annealed to zero, and evaluation runs at zero.
        warchest.set_cap_value(0.0)
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
        snap = self.game.snapshot()
        for occ in snap["board"]:
            assert self.geometry["hex_of"][occ["coord"]] == occ["hex"], occ
        for coord in snap["markers"]:
            assert coord in self.geometry["hex_of"], coord

    def new_game(self):
        seed = self.seed if self.seed is not None else random.getrandbits(63)
        draft = {
            "white_units": STARTER_WHITE,
            "black_units": STARTER_BLACK,
            "first_player": "white",
        }
        self.game = warchest.LiveGame(draft, 1, 0, self.depth, self.iters, seed)
        self.game_id = random.getrandbits(32)

    def snapshot(self):
        snap = self.game.snapshot()
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
                "ckpt": str(s.ckpt),
                "depth": s.depth,
                "iters": s.iters,
            })
        else:
            send_json(self, {"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/move":
            req = read_json(self)
            with LOCK:
                try:
                    snap = SESSION.game.human_move({"code": int(req["code"])})
                except Exception as e:  # illegal action / not your turn
                    send_json(self, {"error": str(e)}, 400)
                    return
                send_json(self, sanitize(snap))
        elif path == "/api/new":
            with LOCK:
                SESSION.new_game()
                send_json(self, SESSION.snapshot())
        else:
            send_json(self, {"error": "not found"}, 404)


def latest_checkpoint() -> Path:
    cands = sorted((ROOT / "runs").glob("*/ckpt_final.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit("no runs/*/ckpt_final.pt found — pass --ckpt explicitly")
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint .pt file (default: newest runs/*/ckpt_final.pt)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--depth", type=int, default=2, help="public-tree depth (train cfg)")
    ap.add_argument("--iters", type=int, default=16, help="CFR iterations (train cfg)")
    ap.add_argument("--seed", type=int, default=None, help="fixed RNG seed for draws")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    global SESSION
    ckpt = args.ckpt or latest_checkpoint()
    print(f"checkpoint: {ckpt}")
    SESSION = Session(ckpt, args.depth, args.iters, args.seed)
    SESSION.check_geometry()
    print(f"you: {', '.join(SESSION.units[u][0] for u in STARTER_WHITE)}")
    print(f"agent: {', '.join(SESSION.units[u][0] for u in STARTER_BLACK)} (black)")
    print(f"agent cfg: depth={args.depth} iters={args.iters}")

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
