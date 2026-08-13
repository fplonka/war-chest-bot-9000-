#!/usr/bin/env python3
"""Newest-first index of runs/, for a file:// browser.

    python3 tools/runs_index.py
    open runs/index.html

`tools/box.sh pull` regenerates this. A run is anything with a log.json.
"""
import datetime
import html
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(RUNS, "index.html")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def mtime(run):
    log = os.path.join(RUNS, run, "log.json")
    try:
        return os.path.getmtime(log)
    except OSError:
        return 0.0


def health(run):
    """One line: schedule, last-epoch fit, throughput, Elo if the ladder ran."""
    log = load_json(os.path.join(RUNS, run, "log.json"))
    if not isinstance(log, dict):
        return ""
    cfg, eps = log.get("cfg") or {}, log.get("epochs") or []
    if not isinstance(cfg, dict):
        cfg = {}
    if not isinstance(eps, list):
        eps = []
    bits = []
    if cfg.get("note"):
        bits.append(cfg["note"])
    if cfg.get("minutes"):
        bits.append(f"{cfg['minutes']:g}m")
    if cfg.get("warm_minutes"):
        bits.append(f"{cfg['warm_minutes']:g} warm")
    git = cfg.get("git")
    if git:
        bits.append(git)
    if cfg.get("seed") is not None:
        bits.append(f"seed {cfg['seed']}")
    last = next((e for e in reversed(eps) if e.get("phase") == "rebel"), None) or (
        eps[-1] if eps else None)
    if last:
        t = last.get("t", 0) / 60
        bits.append(f"t={t:.0f}m")
        if "horizon_frac" in last:
            bits.append(f"horizon {last['horizon_frac']:.0%}")
        sps = last.get("balanced_solves_per_s") or last.get("solves_per_s")
        if sps:
            bits.append(f"{sps:.0f}/s")
        loss, std = last.get("loss"), last.get("tgt_std")
        if loss is not None and std:
            bits.append(f"L/var {loss / max(std * std, 1e-9):.2f}")
    lad = load_json(os.path.join(RUNS, run, "ladder.json")) or {}
    elos = [p for p in lad.get("players") or [] if p.get("t") is not None]
    if elos:
        elos.sort(key=lambda p: p["t"])
        bits.append("elo " + " ".join(f"{p['elo']:+.0f}" for p in elos))
    fin = next((p for p in lad.get("players") or []
                if str(p.get("name", "")).endswith(".final")), None)
    if fin and fin.get("elo") is not None:
        se = fin.get("se") or 0
        bits.append(f"{fin['elo']:+.0f}±{1.96 * se:.0f} (95%) Elo")
    return " · ".join(bits)


def main():
    os.makedirs(RUNS, exist_ok=True)
    names = sorted(
        (n for n in os.listdir(RUNS)
         if os.path.isfile(os.path.join(RUNS, n, "log.json"))),
        key=mtime, reverse=True)
    items = []
    for n in names:
        href = "report.html" if os.path.isfile(os.path.join(RUNS, n, "report.html")) \
            else "train.log" if os.path.isfile(os.path.join(RUNS, n, "train.log")) \
            else "log.json"
        when = datetime.datetime.fromtimestamp(mtime(n)).strftime("%Y-%m-%d %H:%M")
        sub = health(n)
        items.append(
            f'<li><a href="{html.escape(n)}/{href}">{html.escape(n)}</a>'
            f'<span class="date">{when}</span>'
            + (f'<span class="sub">{html.escape(sub)}</span>' if sub else "")
            + "</li>")
    page = f"""<!doctype html>
<meta charset=utf-8><title>runs</title>
<style>
:root{{--bg:#fff;--ink:#111;--mut:#666;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--ink:#e8e8e8;--mut:#999;--line:#333}}}}
body{{margin:0 auto;padding:24px 16px;max-width:56rem;background:var(--bg);color:var(--ink);
 font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
h1{{font-size:14px;margin:0 0 12px}}
ul{{list-style:none;padding:0;margin:0}}
li{{padding:6px 0;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:4px 14px}}
a{{color:var(--ink);font-weight:700;text-decoration:none}}
a:hover{{text-decoration:underline}}
.date{{color:var(--mut);min-width:9.5rem}}
.sub{{color:var(--mut);flex:1}}
.note{{color:var(--mut);font-size:12px;margin-top:16px}}
</style>
<h1>runs</h1>
<ul>
{chr(10).join(items)}
</ul>
<p class="note">{len(names)} runs with a log · file://{html.escape(OUT)}</p>
"""
    with open(OUT, "w") as f:
        f.write(page)
    print(f"[runs-index] {OUT} ({len(names)} runs)")


if __name__ == "__main__":
    main()
