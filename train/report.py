"""One self-contained HTML page per run, or per comparison.

    python train/report.py runs/dcfr-base            # one run
    python train/report.py runs/dcfr-* -o cmp.html   # arms overlaid

`exp.py` calls this when a run finishes, so looking at a result never means
opening a terminal, and never means a plotting window on a box that has no
display -- which is what the matplotlib version this replaces could not do.

The page carries the four things a run is read from (strength against training
time, the value loss split by row age, the spread of the predictions, and
generation throughput), the data-health counters the logs have always recorded
and nobody ever saw, the ladder's own head-to-head records, and the config
delta that says what this arm actually changed. No external files, no CDN, no
fonts to fetch: one page that can be copied off the box and still render.

The head-to-head table is not decoration. Elo assumes transitivity, and in a
game with any rock-paper-scissors structure an arm can gain rating against the
field while losing to the specific control it is meant to beat. When those two
disagree the direct record is the one that answers the experiment's question.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

W, H, PAD = 460, 190, 38
INK = ["#2a78d6", "#008300", "#e87ba4", "#b8860b", "#7b52ab"]


def read(run):
    """A run's log, ladder and truth score; missing pieces are None."""
    with open(f"{run}/log.json") as f:
        log = json.load(f)
    out = {"name": os.path.basename(run.rstrip("/")), "cfg": log.get("cfg", {}),
           "epochs": [e for e in log["epochs"] if e["phase"] == "rebel"],
           "snaps": log.get("snapshots", []), "ladder": None}
    try:
        with open(f"{run}/ladder.json") as f:
            out["ladder"] = json.load(f)
    except FileNotFoundError:
        pass
    return out


def ema(y, span=12):
    """Per-epoch values are genuinely jumpy -- each epoch trains on a fresh
    sample of a two-million-row buffer -- so the trend is the readable part and
    the scatter behind it is the honest context for it."""
    a, out = 2.0 / (span + 1.0), []
    for v in y:
        out.append(v if not out else a * v + (1 - a) * out[-1])
    return out


def chart(title, ylabel, series, zero=False, hlines=()):
    """One SVG panel. `series` is `(label, xs, ys, smooth)`."""
    pts = [(x, y) for _, xs, ys, _ in series for x, y in zip(xs, ys)]
    pts += [(0, v) for _, v in hlines]
    if not pts:
        return f'<figure><figcaption>{title}</figcaption><p class=none>no data</p></figure>'
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = (0 if zero else min(ys)), max(ys)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1
    sx = lambda x: PAD + (x - x0) / (x1 - x0 or 1) * (W - PAD - 12)
    sy = lambda y: H - PAD - (y - y0) / (y1 - y0) * (H - PAD - 14)

    g = [f'<line class=ax x1={PAD} y1={H - PAD} x2={W - 12} y2={H - PAD}/>']
    for f in (0.0, 0.5, 1.0):
        v = y0 + f * (y1 - y0)
        g.append(f'<line class=grid x1={PAD} y1={sy(v):.1f} x2={W - 12} y2={sy(v):.1f}/>')
        g.append(f'<text class=tick x={PAD - 5} y={sy(v) + 3:.1f}>{fmt(v)}</text>')
    g.append(f'<text class=tick x={PAD} y={H - PAD + 14} text-anchor=start>{fmt(x0)}</text>')
    g.append(f'<text class=tick x={W - 12} y={H - PAD + 14} text-anchor=end>{fmt(x1)}</text>')
    for v, c in zip([v for _, v in hlines], INK[1:]):
        g.append(f'<line class=ref stroke="{c}" x1={PAD} y1={sy(v):.1f} '
                 f'x2={W - 12} y2={sy(v):.1f}/>')
    for i, (label, xa, ya, smooth) in enumerate(series):
        c = INK[i % len(INK)]
        if smooth:
            d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xa, ya))
            g.append(f'<polyline class=dots stroke="{c}" points="{d}"/>')
            ya = ema(ya)
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xa, ya))
        g.append(f'<polyline class=line stroke="{c}" points="{d}"/>')
    keys = "".join(f'<span style="color:{INK[i % len(INK)]}">{s[0]}</span>'
                   for i, s in enumerate(series) if s[0])
    keys += "".join(f'<span style="color:{c}">{n}</span>'
                    for (n, _), c in zip(hlines, INK[1:]))
    return (f'<figure><figcaption>{title}<em>{ylabel}</em></figcaption>'
            f'<svg viewBox="0 0 {W} {H}">{"".join(g)}</svg>'
            f'<div class=key>{keys}</div></figure>')


def fmt(v):
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}"


def panels(runs):
    """The four curves, each overlaying every run."""
    mins = lambda r: [e["t"] / 60.0 for e in r["epochs"]]
    one = len(runs) == 1
    tag = lambda r: "" if one else r["name"]

    elo = []
    hl = []
    for r in runs:
        if not r["ladder"]:
            continue
        snaps = [p for p in r["ladder"]["players"] if p["t"] is not None]
        if snaps:
            elo.append((tag(r) or "snapshot", [p["t"] / 60.0 for p in snaps],
                        [p["elo"] for p in snaps], False))
        for ref in ("greedy", "random"):
            p = next((q for q in r["ladder"]["players"] if q["name"] == ref), None)
            if p and ref not in [n for n, _ in hl]:
                hl.append((ref, p["elo"]))

    out = [chart("Strength against training time", "elo", elo, hlines=hl)]

    if one and "loss_old" in (runs[0]["epochs"] or [{}])[-1]:
        r = runs[0]
        out.append(chart("Value loss by row age", "huber", [
            ("old rows", mins(r), [e["loss_old"] for e in r["epochs"]], True),
            ("fresh rows", mins(r), [e["loss_new"] for e in r["epochs"]], True)]))
    else:
        out.append(chart("Value loss", "huber",
                         [(tag(r), mins(r), [e["loss"] for e in r["epochs"]], True)
                          for r in runs]))

    out.append(chart("Spread of predictions (collapse toward 0 = degenerate)", "std",
                     [(tag(r) or "prediction", mins(r),
                       [e["probe_std"] for e in r["epochs"]], True) for r in runs]
                     + ([("target", mins(runs[0]),
                          [e["tgt_std"] for e in runs[0]["epochs"]], True)] if one else []),
                     zero=True))
    out.append(chart("Generation throughput", "solves/s",
                     [(tag(r), mins(r), [e["solves_per_s"] for e in r["epochs"]], True)
                      for r in runs], zero=True))
    # Rows trained per solve generated. This, not the buffer size, is what
    # governs how hard a run overfits its own replay -- and the old-vs-fresh
    # loss split above is its readout.
    out.append(chart("Replay ratio (rows trained per solve)", "x",
                     [(tag(r), mins(r),
                       [e["steps"] * r["cfg"].get("batch", 1024) / max(e["solves"], 1)
                        for e in r["epochs"]], True) for r in runs], zero=True))
    # The horizon cuts a game at 256 coin plays and scores it a draw, and War
    # Chest has no draws. A rising rate means the ladder below is measuring a
    # game that is increasingly not the real one.
    out.append(chart("Games cut at the horizon", "fraction",
                     [(tag(r), mins(r), [e["horizon_frac"] for e in r["epochs"]], True)
                      for r in runs], zero=True))
    return out


def health(r):
    e = r["epochs"]
    if not e:
        return ""
    last, tot = e[-1], lambda k: sum(x.get(k, 0) for x in e)
    cells = [("wall clock", f"{last['t'] / 60:.0f} min"),
             ("solves", f"{tot('solves'):,}"),
             ("solves/s", f"{last['solves_per_s']:.0f}"),
             ("buffer", f"{last['buf']:,}"),
             ("node-cap fallbacks", f"{tot('node_caps'):,}"),
             ("dropped solves", f"{tot('dropped'):,}"),
             ("exact CPU fallbacks", f"{tot('exact_fallbacks'):,}"),
             ("games cut at horizon", f"{last['horizon_frac']:.1%}")]
    return "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in cells)


def ladder_table(lad):
    if not lad:
        return "<p class=none>no ladder yet — <code>python train/ladder.py &lt;run&gt;</code></p>"
    def prow(p):
        when = "—" if p["t"] is None else f"{p['t'] / 60:.0f} min"
        return (f"<tr><td>{p['name']}</td><td class=n>{when}</td>"
                f"<td class=n>{p['elo']:.0f}</td><td class=n>±{p['se']:.0f}</td>"
                f"<td class=n>{p['score']:.3f}</td></tr>")

    rows = "".join(prow(p) for p in sorted(lad["players"], key=lambda p: -p["elo"]))
    pairs = "".join(
        f"<tr><td>{p['a']} <span class=vs>vs</span> {p['b']}</td>"
        f"<td class=n>{p['w']}–{p['l']}–{p['d']}</td>"
        f"<td class=n>{p.get('n', p['w'] + p['l'] + p['d'])}</td>"
        f"<td class=n>{p['score']:.3f}</td></tr>"
        for p in sorted(lad["pairs"], key=lambda p: -(p.get("n", 0))))
    return (f"<table><thead><tr><th>player<th class=n>trained<th class=n>elo"
            f"<th class=n>±<th class=n>score</thead>{rows}</table>"
            f"<h3>Head to head</h3><table><thead><tr><th>pairing<th class=n>W–L–D"
            f"<th class=n>games<th class=n>score</thead>{pairs}</table>")


CSS = """
:root{--ink:#0b0b0b;--mut:#52514e;--line:#e6e5e2;--bg:#fcfcfb}
*{box-sizing:border-box}
body{margin:0;padding:32px;background:var(--bg);color:var(--ink);
 font:14px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px}
h1{font-size:19px;margin:0 0 2px}h2{font-size:15px;margin:30px 0 10px}
h3{font-size:13px;color:var(--mut);margin:20px 0 6px;font-weight:600}
.sub{color:var(--mut);margin:0 0 4px}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.delta{margin:10px 0 0;padding:0}
.delta span{display:inline-block;background:#eef3fb;color:#1c4e8a;border-radius:4px;
 padding:2px 7px;margin:0 6px 6px 0;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.delta span.none{background:var(--line);color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}
figure{margin:0;border:1px solid var(--line);border-radius:8px;padding:10px 6px 6px;
 background:#fff;overflow:hidden}
figcaption{font-size:12px;font-weight:600;padding:0 8px 4px}
figcaption em{color:var(--mut);font-weight:400;font-style:normal;float:right}
svg{width:100%;height:auto;display:block}
.line{fill:none;stroke-width:2}.dots{fill:none;stroke-width:1;opacity:.16}
.ax,.grid line{stroke:var(--line)}line.grid{stroke:var(--line)}
.ref{stroke-width:1.3;stroke-dasharray:4 3}
.tick{font-size:9px;fill:var(--mut);text-anchor:end}
.key{font-size:11px;padding:2px 8px 0}.key span{margin-right:12px;font-weight:600}
dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;
 margin:10px 0 0;padding:0}
dt{font-size:11px;color:var(--mut)}dd{margin:0;font-size:15px;
 font-family:ui-monospace,Menlo,monospace}
table{border-collapse:collapse;width:100%;margin:6px 0;font-size:13px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}
th{font-size:11px;color:var(--mut);font-weight:600}
td.n,th.n{text-align:right;font-family:ui-monospace,Menlo,monospace}
.vs{color:var(--mut)}.none{color:var(--mut);font-size:13px}
footer{color:var(--mut);font-size:12px;margin-top:28px;border-top:1px solid var(--line);
 padding-top:10px}
"""


def page(runs, title):
    d = [f"<span>{k}={v}</span>" for k, v in config.delta(runs[0]["cfg"]).items()] \
        if len(runs) == 1 else \
        [f"<span>{r['name']}: " + (", ".join(f"{k}={v}" for k, v in
                                             config.delta(r["cfg"]).items()) or "baseline")
         + "</span>" for r in runs]
    body = [f"<h1>{title}</h1>",
            f"<p class=sub mono>{runs[0]['cfg'].get('git', '?')} · "
            f"{len(runs[0]['epochs'])} ReBeL epochs</p>",
            f"<div class=delta>{''.join(d) or '<span class=none>baseline</span>'}</div>",
            "<h2>Curves</h2>", f"<div class=grid>{''.join(panels(runs))}</div>"]
    for r in runs:
        body.append(f"<h2>{r['name']}</h2><dl>{health(r)}</dl>")
        body.append(ladder_table(r["ladder"]))
    body.append("<footer>Generated by train/report.py. Elo error bars are per-player "
                "placement, not a test between two players; a difference is only "
                "resolved when the pairing between them has the games to resolve it "
                "(≈1,000 games for 22 Elo).</footer>")
    return (f"<!doctype html><meta charset=utf-8><title>{title}</title>"
            f"<style>{CSS}</style>{''.join(body)}")


def write(runs, out=None, title=None):
    """Render `runs` (directories) to one page. The callable form exp.py uses."""
    loaded = [read(r) for r in runs]
    out = out or f"{runs[0]}/report.html"
    with open(out, "w") as f:
        f.write(page(loaded, title or " · ".join(r["name"] for r in loaded)))
    print(f"[report] {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", default=None,
                    help="output path (default: <first run>/report.html)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    write(args.runs, args.out, args.title)


if __name__ == "__main__":
    main()
