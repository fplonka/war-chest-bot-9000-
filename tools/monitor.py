#!/usr/bin/env python3
"""Live view of runs/, read from disk on every request.

    python3 tools/monitor.py                    # http://127.0.0.1:8420
    python3 tools/monitor.py --pull

There is no generated page on disk and no regeneration step: a run in progress
is a log.json with fewer epochs in it than it will have in a minute, and it
renders exactly like a finished one.
"""
import argparse
import glob
import json
import math
import os
import shlex
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
LIVE = 120    # a log.json untouched for this long is not a live run


def read_json(path):
    """A JSON object, or None. An rsync caught mid-transfer hands us half a
    file, and the oldest runs wrote log.json as a bare list of epochs; neither
    is worth a 500, and neither has anything to draw."""
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


def panels(eps, elo):
    """Every panel of the dashboard, from the SoG epochs logged so far."""
    m = [e["t"] / 60.0 for e in eps]
    col = lambda k: [e.get(k) for e in eps]
    has = lambda k: any(k in e for e in eps)
    out = [elo]

    if has("loss_old"):
        out.append(panel("Value loss by row age", "huber", m,
                         [series("training sample", col("loss"), True),
                          series("old rows", col("loss_old"), True),
                          series("fresh rows", col("loss_new"), True)]))
    else:
        out.append(panel("Value loss", "huber", m,
                         [series("value", col("loss"), True)]))

    if has("policy_loss"):
        out.append(panel("Policy loss", "cross-entropy", m,
                         [series("policy", col("policy_loss"), True)], zero=True))
    if has("policy_weighted_loss"):
        out.append(panel("Per-head objective", "weighted loss", m,
                         [series("value Huber", col("loss"), True),
                          series("policy", col("policy_weighted_loss"), True),
                          series("total", col("total_loss"), True)], zero=True))
    if has("policy_target_entropy"):
        out.append(panel("Policy entropy", "nats / information state", m,
                         [series("network prior", col("policy_prior_entropy"), True),
                          series("search target", col("policy_target_entropy"), True)],
                         zero=True))
        out.append(panel("Stored search target vs current prior", "KL divergence (nats)", m,
                         [series("target KL", col("policy_search_kl"), True)], zero=True))

    # Historical runs may carry the ablated ownership head's metrics.
    if has("aux_loss"):
        out.append(panel("Auxiliary ownership loss", "cross-entropy", m,
                         [series("ownership", col("aux_loss"), True)], zero=True))
        out.append(panel("Auxiliary ownership accuracy", "fraction correct", m,
                         [series("ownership", col("aux_acc"), True)], zero=True,
                         hlines=[("chance", 1 / 3)]))

    relative_loss = [e["loss"] / e["tgt_std"] ** 2
                     if e.get("tgt_std") and "loss" in e else None for e in eps]
    out.append(panel("Relative value loss", "huber / target variance", m,
                     [series("value", relative_loss, True)], zero=True))
    out.append(panel("Spread of predictions", "std", m,
                     [series("prediction", col("probe_std"), True),
                      series("target configs", col("tgt_std"), True),
                      series("belief-weighted target", col("tgt_belief_std"), True)],
                     zero=True))
    if has("tgt_p05"):
        out.append(panel("Value targets", "target", m,
                         [series("p05", col("tgt_p05"), True),
                          series("median", col("tgt_p50"), True),
                          series("p95", col("tgt_p95"), True),
                          series("belief mean", col("tgt_belief_mean"), True)],
                         hlines=[("zero", 0)]))
    if has("value_outcome_rmse"):
        out.append(panel("Value head against game outcomes", "error", m,
                         [series("RMSE", col("value_outcome_rmse"), True),
                          series("MAE", col("value_outcome_mae"), True)], zero=True))
        out.append(panel("Value calibration", "statistic", m,
                         [series("correlation", col("value_outcome_corr"), True),
                          series("slope", col("value_calibration_slope"), True),
                          series("bias", col("value_outcome_bias"), True)],
                         hlines=[("ideal slope", 1), ("zero", 0)]))

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
                      series("run average", col("solves_per_s"))], zero=True))
    if has("buf"):
        out.append(panel("Replay buffer fill", "rows", m,
                         [series("buffer", col("buf"))], zero=True))
    if has("replay_query_frac"):
        out.append(panel("Replay composition", "fraction of live rows", m,
                         [series("warm", col("replay_warm_frac"), True),
                          series("main line", col("replay_play_frac"), True),
                          series("query", col("replay_query_frac"), True),
                          series("TD(1) rows", col("replay_td1_row_frac"), True),
                          series("TD(1) target cells", col("replay_td1_target_frac"), True)],
                         zero=True))
        out.append(panel("Sampled replay composition", "fraction of sampled rows", m,
                         [series("warm", col("sample_warm_frac"), True),
                          series("main line", col("sample_play_frac"), True),
                          series("query", col("sample_query_frac"), True),
                          series("TD(1) target cells", col("sample_td1_target_frac"), True)],
                         zero=True))
    if has("sample_age_mean"):
        out.append(panel("Target age when sampled", "seconds", m,
                         [series("mean", col("sample_age_mean"), True),
                          series("median", col("sample_age_p50"), True),
                          series("p90", col("sample_age_p90"), True),
                          series("oldest target", col("target_age_max"), True),
                          series("oldest insertion", col("buf_s"), True)], zero=True))
        out.append(panel("Target delivery delay", "seconds", m,
                         [series("p90", col("sample_delay_p90"), True),
                          series("warm mean", col("sample_warm_delay"), True),
                          series("main-line mean", col("sample_play_delay"), True),
                          series("query mean", col("sample_query_delay"), True)], zero=True))
    if has("grad_norm"):
        out.append(panel("Gradient norm before clipping", "L2 norm", m,
                         [series("mean", col("grad_norm"), True),
                          series("max", col("grad_norm_max"), True)], zero=True,
                         hlines=[("clip", 5)]))
        out.append(panel("Weight norm", "L2 norm", m,
                         [series("weights", col("weight_norm"), True)], zero=True))
    if has("zero_sum_rms"):
        out.append(panel("Zero-sum residual", "|E[v0] + E[v1]|", m,
                         [series("RMS", col("zero_sum_rms"), True),
                          series("batch max", col("zero_sum_max"), True)], zero=True))

    for title, ylabel, key, smooth in (
            ("Replay generation throughput", "rows/s", "rows_per_s", False),
            ("Effective training ratio", "optimizer rows / solve",
             "effective_train_ratio", False),
            ("Passes per generated row", "optimizer rows / replay row",
             "train_row_ratio", False),
            ("Gradient clipping", "fraction of steps", "grad_clip_frac", True),
            ("Replay age", "seconds retained", "buf_s", False),
            # The horizon cuts a game at 256 coin plays and scores it a draw,
            # and War Chest has no draws. A rising rate means the ladder below
            # is measuring a game that is increasingly not the real one.
            ("Games cut at horizon", "fraction", "horizon_frac", True)):
        if has(key):
            out.append(panel(title, ylabel, m,
                             [series(title.lower(), col(key), smooth)], zero=True))

    if any(e.get("plays") for e in eps):
        kinds = ("attack", "maneuver", "deploy", "bolster", "recruit",
                 "pass", "claim_initiative")
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
    if reasons:
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
           ("dropped solves", f"{tot('dropped'):,}"),
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
    """One line per run. Re-reading 150 log.json files every three seconds is
    the only thing here worth avoiding, and an mtime settles it."""
    out = []
    for path in glob.glob(os.path.join(runs_dir, "*", "log.json")):
        name, mt = os.path.basename(os.path.dirname(path)), os.path.getmtime(path)
        hit = SUMMARY.get(name)
        if not hit or hit[0] != mt:
            log = read_json(path) or {}
            eps = log.get("epochs") or []
            last = eps[-1] if eps else {}
            hit = (mt, {"name": name, "mtime": mt, "epochs": len(eps),
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
    # curve to the floor in its final pixel.
    eps = [e for e in log.get("epochs") or []
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


def puller(src, dest, every=30):
    """Keep the laptop current while serving it. rsync failures are ordinary
    here -- the box reboots, a run replaces log.json mid-transfer (exit 24) --
    and none of them is a reason to stop serving what is already on disk."""
    while True:
        r = subprocess.run(["rsync", "-az", "--exclude", "*.pt", "--exclude", "*.tmp",
                            src, dest + os.sep], capture_output=True, text=True)
        if r.returncode:
            print(f"[monitor] pull {r.returncode}: {r.stderr.strip()[:200]}", flush=True)
        time.sleep(every)


def main():
    host = os.environ.get("WARCHEST_BOX_HOST", "ssh1.vast.ai")
    port = os.environ.get("WARCHEST_BOX_PORT", "26778")
    key = os.path.expanduser(os.environ.get(
        "WARCHEST_BOX_KEY", "~/.ssh/id_ed25519_warchest_vast"))
    remote = os.environ.get("WARCHEST_BOX_DIR", "/workspace/warchest-engine")
    pull_default = f"root@{host}:{remote}/runs/"
    rsh = shlex.join([
        "ssh", "-i", key, "-p", port, "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30"])

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--arena", default=os.path.join(HERE, "arena"))
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--pull", nargs="?", const=pull_default, metavar="SRC",
                    help="pull every 30s; omit SRC for the WARCHEST_BOX_* box")
    args = ap.parse_args()
    if args.pull:
        os.environ.setdefault("RSYNC_RSH", rsh)
        threading.Thread(target=puller, args=(args.pull, args.runs),
                         daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.runs = os.path.abspath(args.runs)
    srv.arena = os.path.abspath(args.arena)
    print(f"[monitor] http://127.0.0.1:{args.port} · {srv.runs}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
