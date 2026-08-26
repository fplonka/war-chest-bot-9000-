#!/usr/bin/env python3
"""Live view of runs/, read from disk on every request.

    python3 tools/monitor.py                    # http://127.0.0.1:8420
    python3 tools/monitor.py --pull

There is no generated page on disk and no regeneration step: a run in progress
appends epochs.jsonl and renders exactly like a finished one.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(HERE, "tools")
PAGE = os.path.join(TOOLS, "monitor.html")
ASSETS = {
    "/vendor/uPlot.iife.min.js": ("uPlot.iife.min.js", "text/javascript"),
    "/vendor/uPlot.min.css": ("uPlot.min.css", "text/css"),
}
sys.path.insert(0, os.path.join(HERE, "train"))
import config  # noqa: E402  -- knobs(), so the baseline lives in one place

CI95 = 1.96   # ladder.json stores the 1-sigma Bradley-Terry SE
LIVE = 120    # an epoch log untouched for this long is not a live run
EPOCH_LIMIT = 2048


def read_json(path):
    """A complete JSON object, or None if a pull caught it mid-write."""
    try:
        with open(path) as f:
            got = json.load(f)
        return got if isinstance(got, dict) else None
    except (OSError, ValueError):
        return None


def read_text(path, limit=64 << 10):
    """The tail of a file, or "" if there is none."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - limit))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def jsonl_record(raw):
    try:
        got = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return None
    return got if isinstance(got, dict) else None


def last_jsonl(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - (64 << 10)))
            lines = f.read().splitlines()
    except OSError:
        return {}
    for raw in reversed(lines):
        got = jsonl_record(raw)
        if got is not None:
            return got
    return {}


def read_epochs(path, limit=EPOCH_LIMIT):
    """Read all small logs, or evenly sample a large JSONL log by byte offset."""
    if limit <= 0:
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if not size:
                return []
            if size <= (8 << 20):
                f.seek(0)
                return [got for raw in f
                        if (got := jsonl_record(raw)) is not None]
            out, starts = [], set()
            for i in range(limit):
                offset = size * i // limit
                f.seek(offset)
                if offset:
                    f.readline()
                start = f.tell()
                got = jsonl_record(f.readline())
                if got is not None and start not in starts:
                    starts.add(start)
                    out.append(got)
            f.seek(max(0, size - (64 << 10)))
            last = next((got for raw in reversed(f.read().splitlines())
                         if (got := jsonl_record(raw)) is not None), None)
            if last is not None:
                if len(out) == limit:
                    out[-1] = last
                elif not out or out[-1] != last:
                    out.append(last)
            return out
    except OSError:
        return []


def finite(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def series(label, y, smooth=False, err=None):
    return {"n": label, "y": [finite(v) for v in y], "sm": smooth,
            "e": [finite(v) for v in err] if err else None}


def panel(title, ylabel, x, ss, zero=False, hlines=(), marks=False):
    """Return None until at least one series has data."""
    ss = [s for s in ss if any(v is not None for v in s["y"])]
    if not ss:
        return None
    return {"t": title, "y": ylabel, "x": [finite(v) for v in x], "s": ss,
            "zero": zero, "h": list(hlines), "marks": marks}


def elo_panel(lad, name):
    """This run's snapshots on a ladder that may also hold other bots. A
    snapshot bot is named `<run>.<label>` and carries the minutes it trained
    for; anything else on the ladder is a reference, not a point on the curve."""
    if not lad:
        return None
    ps = sorted((p for p in lad.get("players", [])
                 if p.get("minutes") is not None
                 and p["name"].startswith(name + ".")),
                key=lambda p: p["minutes"])
    greedy = next((p for p in lad.get("players", []) if p["name"] == "greedy"), None)
    x = [p["minutes"] for p in ps]
    return panel("Strength vs training time", "elo (95% CI)", x,
                 [series("snapshot", [p["elo"] for p in ps],
                         err=[CI95 * (p.get("se") or 0) for p in ps])],
                 hlines=[("greedy", greedy["elo"])] if greedy else (), marks=True)


PANELS = (
    ("Value loss", "huber", (("value", "loss"),), False, ()),
    ("Value loss by row age", "huber",
     (("old rows", "loss_old"), ("fresh rows", "loss_new")), False, ()),
    ("Policy loss", "cross-entropy", (("policy", "policy_loss"),), True, ()),
    ("Per-head objective", "weighted loss",
     (("policy", "policy_weighted_loss"), ("total", "total_loss")), True, ()),
    ("Policy entropy", "nats / information state",
     (("network prior", "policy_prior_entropy"),
      ("search target", "policy_target_entropy")), True, ()),
    ("Stored search target vs current prior", "KL divergence (nats)",
     (("target KL", "policy_search_kl"),), True, ()),
    ("Auxiliary ownership loss", "cross-entropy", (("ownership", "aux_loss"),), True, ()),
    ("Auxiliary ownership accuracy", "fraction correct", (("ownership", "aux_acc"),),
     True, (("chance", 1 / 3),)),
    ("Spread of predictions", "std",
     (("prediction", "probe_std"), ("target configs", "tgt_std"),
      ("belief-weighted target", "tgt_belief_std")), True, ()),
    ("Value targets", "target",
     (("p05", "tgt_p05"), ("median", "tgt_p50"), ("p95", "tgt_p95"),
      ("belief mean", "tgt_belief_mean")), False, (("zero", 0),)),
    ("Value head against game outcomes", "error",
     (("RMSE", "value_outcome_rmse"), ("MAE", "value_outcome_mae")), True, ()),
    ("Value calibration", "statistic",
     (("correlation", "value_outcome_corr"), ("slope", "value_calibration_slope"),
      ("bias", "value_outcome_bias")), False, (("ideal slope", 1), ("zero", 0))),
    ("Replay buffer fill", "rows", (("buffer", "buf"),), True, ()),
    ("Replay composition", "fraction of live rows",
     (("warm", "replay_warm_frac"), ("main line", "replay_play_frac"),
      ("query", "replay_query_frac"), ("TD(1) rows", "replay_td1_row_frac"),
      ("TD(1) target cells", "replay_td1_target_frac")), True, ()),
    ("Sampled replay composition", "fraction of sampled rows",
     (("warm", "sample_warm_frac"), ("main line", "sample_play_frac"),
      ("query", "sample_query_frac"), ("TD(1) target cells", "sample_td1_target_frac")),
     True, ()),
    ("Target age when sampled", "seconds",
     (("mean", "sample_age_mean"), ("median", "sample_age_p50"),
      ("p90", "sample_age_p90"), ("oldest target", "target_age_max")), True, ()),
    ("Target delivery delay", "seconds",
     (("p90", "sample_delay_p90"), ("warm mean", "sample_warm_delay"),
      ("main-line mean", "sample_play_delay"), ("query mean", "sample_query_delay")),
     True, ()),
    ("Gradient norm before clipping", "L2 norm",
     (("mean", "grad_norm"), ("max", "grad_norm_max")), True, (("clip", 5),)),
    ("Weight norm", "L2 norm", (("weights", "weight_norm"),), True, ()),
    ("Zero-sum residual", "|E[v0] + E[v1]|",
     (("RMS", "zero_sum_rms"), ("batch max", "zero_sum_max")), True, ()),
    ("Replay generation throughput", "rows/s", (("replay generation throughput", "rows_per_s"),),
     True, ()),
    ("Effective training ratio", "optimizer rows / solve",
     (("effective training ratio", "effective_train_ratio"),), True, ()),
    ("Passes per generated row", "optimizer rows / replay row",
     (("passes per generated row", "train_row_ratio"),), True, ()),
    ("Gradient clipping", "fraction of steps", (("gradient clipping", "grad_clip_frac"),),
     True, ()),
    ("Replay age", "seconds retained", (("replay age", "buf_s"),), True, ()),
    # The horizon cuts a game at 256 coin plays and scores it a draw. A rising
    # rate means the ladder is measuring a game that is increasingly not real.
    ("Games cut at horizon", "fraction", (("games cut at horizon", "horizon_frac"),),
     True, ()),
)


def panels(eps, elo):
    """Every panel of the dashboard, from the SoG epochs logged so far."""
    m = [e["t"] / 60.0 for e in eps]
    out = [elo] + [
        panel(title, ylabel, m,
              [series(label, [e.get(key) for e in eps], True)
               for label, key in specs], zero=zero, hlines=hlines)
        for title, ylabel, specs, zero, hlines in PANELS]

    relative_loss = [e["loss"] / e["tgt_std"] ** 2
                     if e.get("tgt_std") and "loss" in e else None for e in eps]
    out.append(panel("Relative value loss", "huber / target variance", m,
                     [series("value", relative_loss, True)], zero=True))

    # Two throughput lines, and the gap between them is the information.
    # `solves_per_s` is cumulative solves over elapsed -- the run average, not
    # the current rate -- so "now" has to come from consecutive epochs.
    now = [None]
    for a, b in zip(eps, eps[1:]):
        dt = b["t"] - a["t"]
        now.append(b.get("solves") / dt if dt > 0.5 and b.get("solves") is not None
                   else None)
    out.append(panel("Generation throughput", "solves/s", m,
                     [series("now", now, True),
                      series("run average", [e.get("solves_per_s") for e in eps])],
                     zero=True))

    kinds = ("attack", "maneuver", "deploy", "bolster", "recruit", "pass",
             "claim_initiative")
    kinds = [kind for kind in kinds
             if any(kind in (e.get("plays") or {}) for e in eps)]
    out.append(panel("Move mix", "% of decisions", m,
                     [series(kind.replace("_", " "),
                             [100 * e.get("plays", {}).get(kind, 0)
                              / max(e.get("decisions", 1), 1) for e in eps], True)
                      for kind in kinds], zero=True))

    censuses = [e.get("stop_census") or {} for e in eps]
    reasons = sorted({reason for census in censuses for stops in census.values()
                      for reason in stops})

    def stop_share(census, reason):
        groups = [stops for stops in census.values() if isinstance(stops, dict)]
        total = sum(item.get("count", 0) for stops in groups
                    for item in stops.values())
        count = sum(stops.get(reason, {}).get("count", 0) for stops in groups)
        return 100 * count / total if total else None

    out.append(panel("Solver stop mix", "% of solves", m,
                     [series(reason.removeprefix("budget_"),
                             [stop_share(c, reason) for c in censuses], True)
                      for reason in reasons], zero=True))
    return [p for p in out if p]


def health(eps):
    if not eps:
        return []
    last, tot = eps[-1], lambda k: sum(e.get(k, 0) for e in eps)
    # The last window of a run drains with no games in it, so its horizon
    # fraction is 0.0 by definition; the run-level figure is the cumulative
    # fraction, the same number train.py prints in its gpu-summary line.
    horizon_games = max(tot("games"), 1)
    horizon = 100 * sum(e.get("horizon_frac", 0) * e.get("games", 0)
                        for e in eps) / horizon_games
    out = [("wall clock", f"{last.get('t', 0) / 60:.0f} min"),
           ("solves", f"{tot('solves'):,}"),
           ("solves/s", f"{last.get('solves_per_s', 0):.0f}"),
           ("buffer", f"{last.get('buf', 0):,}"),
           ("dropped queries", f"{tot('dropped'):,}"),
           ("games cut at horizon", f"{horizon:.1f}%")]
    if "aux_loss" in last:
        out.append(("aux ownership ce / accuracy",
                    f"{last['aux_loss']:.3f} / {last['aux_acc']:.1%}"))
    if "effective_train_ratio" in last:
        out += [("effective train ratio", f"{last['effective_train_ratio']:.3f} /solve"),
                ("passes per row", f"{last.get('train_row_ratio', 0):.3f}"),
                ("gradient norm / clipped",
                 f"{last.get('grad_norm', 0):.2f} / {last.get('grad_clip_frac', 0):.1%}"),
                ("weight norm", f"{last.get('weight_norm', 0):.1f}"),
                ("value outcome RMSE / corr",
                 f"{last.get('value_outcome_rmse', 0):.3f} / "
                 f"{last.get('value_outcome_corr', 0):.3f}"),
                ("search KL from prior", f"{last.get('policy_search_kl', 0):.3f}"),
                ("replay warm / play / query",
                 f"{last.get('replay_warm_frac', 0):.0%} / "
                 f"{last.get('replay_play_frac', 0):.0%} / "
                 f"{last.get('replay_query_frac', 0):.0%}"),
                ("target age p90 / oldest",
                 f"{last.get('sample_age_p90', 0) / 60:.1f} / "
                 f"{last.get('target_age_max', 0) / 60:.1f} min"),
                ("target delivery delay p90",
                 f"{last.get('sample_delay_p90', 0) / 60:.1f} min"),
                ("zero-sum RMS / max",
                 f"{last.get('zero_sum_rms', 0):.2e} / "
                 f"{last.get('zero_sum_max', 0):.2e}")]
    return out


SUMMARY = {}   # name -> (mtime, entry): the mtime check is the whole cache


def index(runs_dir):
    """One line per run. The epoch log mtime is the live-run signal."""
    out = []
    for path in glob.glob(os.path.join(runs_dir, "*", "log.json")):
        name = os.path.basename(os.path.dirname(path))
        epoch_path = os.path.join(os.path.dirname(path), "epochs.jsonl")
        try:
            epoch_mt = os.path.getmtime(epoch_path)
        except OSError:
            epoch_mt = 0
        mt = max(os.path.getmtime(path), epoch_mt)
        hit = SUMMARY.get(name)
        if not hit or hit[0] != mt:
            log = read_json(path) or {}
            last = last_jsonl(epoch_path)
            epoch = last.get("epoch", -1)
            hit = (mt, {"name": name, "mtime": mt,
                        "epochs": int(epoch) + 1 if isinstance(epoch, (int, float)) else 0,
                        "phase": last.get("phase", ""),
                        "minutes": round(last.get("t", 0) / 60.0),
                        "note": (log.get("cfg") or {}).get("note", "")})
            SUMMARY[name] = hit
        out.append(dict(hit[1], running=time.time() - mt < LIVE))
    return sorted(out, key=lambda r: -r["mtime"])


def detail(runs_dir, name):
    path = os.path.join(runs_dir, name)
    if not os.path.isdir(path):
        return None
    log = read_json(os.path.join(path, "log.json")) or {}
    # Epochs that generated nothing are the drain at the end of a run: they
    # report loss 0 and a handful of solves, and plotting them drags every
    # curve to the floor in its final pixel. Large logs are sampled before
    # parsing, so a dashboard request never loads the whole run.
    eps = [e for e in read_epochs(os.path.join(path, "epochs.jsonl"))
           if e.get("phase") == "sog" and e.get("solves", 0) > 0
           and e.get("steps", 1) > 0]
    cfg = log.get("cfg") or {}
    # Only ladders in the current format. Older runs kept a `ladder.json`
    # written by code that no longer exists; the file is history, not something
    # to render.
    lads = {os.path.basename(p)[:-5]: read_json(p)
            for p in sorted(glob.glob(os.path.join(path, "ladder*.json")))}
    lads = {k: v for k, v in lads.items() if v and v.get("kind") == "ladder"}
    out = {"name": name, "epochs": len(eps),
           "cfg": [[k, str(v), ch] for k, v, ch in config.knobs(cfg)],
           "panels": panels(eps, elo_panel(lads.get("ladder")
                                           or next(iter(lads.values()), None), name)),
           "health": health(eps),
           "snaps": [s.get("t", 0) / 60.0 for s in log.get("snapshots") or []]}
    for key, val in (("note", cfg.get("note")), ("git", cfg.get("git")),
                     ("ladders", lads), ("log", read_text(f"{path}/train.log")),
                     ("notes", read_text(f"{path}/NOTES.md"))):
        if val:
            out[key] = val
    return out


def arena_summary(report):
    """The one line that says what a report found, whichever kind it is."""
    if report["kind"] == "ladder":
        players = report.get("players") or [{"name": "?"}]
        best = max(players, key=lambda p: p["elo"] if p.get("elo") is not None else -1e9)
        return f"{len(players)} bots · {report.get('games')} games · top {best['name']}"
    best = max(report["bots"], key=lambda b: b["rate"])
    return (f"{len(report['bots'])} bots · {report['questions']} proven "
            f"positions · top {best['bot']}")


def arena_index(arena_dir):
    """One line per report in arena/, newest first."""
    out = []
    for path in glob.glob(os.path.join(arena_dir, "*.json")):
        report = read_json(path)
        if not report or report.get("kind") not in ("ladder", "tablebase"):
            continue
        out.append({"name": os.path.basename(path)[:-5],
                    "mtime": os.path.getmtime(path),
                    "kind": report["kind"],
                    "sub": arena_summary(report)})
    return sorted(out, key=lambda a: -a["mtime"])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?")[0]
        if route == "/":
            return self.body(read_text(PAGE, 1 << 20).encode(), "text/html")
        if route in ASSETS:
            filename, ctype = ASSETS[route]
            try:
                with open(os.path.join(TOOLS, "vendor", filename), "rb") as f:
                    raw = f.read()
            except OSError:
                return self.send_error(404)
            return self.body(raw, ctype)
        if route == "/api/runs":
            return self.body(json.dumps(index(self.server.runs)).encode(),
                             "application/json")
        if route == "/api/arena":
            return self.body(json.dumps(arena_index(self.server.arena)).encode(),
                             "application/json")
        if route.startswith("/api/arena/"):
            name = unquote(route[len("/api/arena/"):])
            got = self.safe_name(name) and read_json(
                os.path.join(self.server.arena, name + ".json"))
            return (self.body(json.dumps(got).encode(), "application/json")
                    if got else self.send_error(404))
        if route.startswith("/api/run/"):
            name = unquote(route[len("/api/run/"):])
            if not self.safe_name(name):
                return self.send_error(404)
            got = detail(self.server.runs, name)
            return (self.body(json.dumps(got).encode(), "application/json")
                    if got else self.send_error(404))
        if route.startswith("/run/") and route.endswith("/train.log"):
            name = unquote(route[len("/run/"):-len("/train.log")])
            if not self.safe_name(name):
                return self.send_error(404)
            try:
                with open(os.path.join(self.server.runs, name, "train.log"), "rb") as f:
                    raw = f.read()
            except OSError:
                return self.send_error(404)
            return self.body(raw, "text/plain; charset=utf-8")
        self.send_error(404)

    @staticmethod
    def safe_name(name):
        return os.sep not in name and name not in ("", ".", "..") and not name.startswith(".")

    def body(self, raw, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass          # a poll every three seconds is not news


def puller(dest, every=30):
    """Keep the laptop current while serving it through box.sh."""
    env = os.environ.copy()
    env["WARCHEST_BOX_LOCAL_DIR"] = os.path.abspath(dest)
    command = [os.path.join(TOOLS, "box.sh"), "pull"]
    while True:
        r = subprocess.run(command, env=env, capture_output=True, text=True)
        if r.returncode:
            print(f"[monitor] pull {r.returncode}: {r.stderr.strip()[:200]}", flush=True)
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--arena", default=os.path.join(HERE, "arena"))
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--pull", action="store_true",
                    help="pull runs from the box every 30s")
    args = ap.parse_args()
    if args.pull:
        threading.Thread(target=puller, args=(args.runs,), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.runs = os.path.abspath(args.runs)
    srv.arena = os.path.abspath(args.arena)
    print(f"[monitor] http://127.0.0.1:{args.port} · {srv.runs}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
