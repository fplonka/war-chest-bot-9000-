"""One HTML page per run, or several overlaid.

    python train/report.py runs/base
    python train/report.py runs/a runs/b -o cmp.html
"""

import argparse
import html
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots

import config

# Series colours. Mid-tone and saturated, so one rendering reads on a light or a
# dark page -- the plot ground is transparent and the page's own background
# shows through, so these have to work against both.
INK = ["#3d8bfd", "#20a37a", "#e0709f", "#d99b28", "#9b7bd4"]
AXIS = "#8a8a8a"
JS = "plotly.min.js"        # written once into runs/, shared by every report
RUNS_DIR = "runs"
# Plot and table show 95% CIs; ladder.json stores 1σ Bradley-Terry SE.
CI95 = 1.96


def ci95(se):
    return CI95 * se if isinstance(se, (int, float)) and math.isfinite(se) else 0


def write_text(path, value):
    """Publish a complete report without exposing a partially written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(value)
    os.replace(tmp, path)


def read(run):
    """A run's log and ladder; missing pieces are None."""
    with open(f"{run}/log.json") as f:
        log = json.load(f)
    # Epochs that generated nothing are the drain at the end of a run: they
    # report loss 0 and a handful of solves, and plotting them drags every
    # curve to the floor in its final pixel.
    out = {"name": os.path.basename(run.rstrip("/")), "path": run.rstrip("/"),
           "cfg": log.get("cfg", {}),
           "epochs": [e for e in log["epochs"]
                      if e["phase"] == "rebel" and e.get("solves", 0) > 0
                      and e.get("steps", 1) > 0],
           "snaps": log.get("snapshots", []), "ladder": None,
           # The warm phase is not plotted -- different objective, different
           # target scale -- but how well it fitted decides how expensive every
           # ReBeL solve is, so its last epoch is kept for the summary.
           "warm_end": next((e for e in reversed(log["epochs"])
                             if e["phase"] == "greedy"), None)}
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
    a, out, prev = 2.0 / (span + 1.0), [], None
    for v in y:
        if not math.isfinite(v):
            out.append(None)
            continue
        prev = v if prev is None else a * v + (1 - a) * prev
        out.append(prev)
    return out


def panel(title, ylabel, series, zero=False, hlines=(), markers=False, yerr=None):
    """A panel's specification. Drawn later, into a shared subplot grid.

    This used to be a hand-rolled SVG emitter -- axis lines, tick placement and
    number formatting all written here, and all of it silently broken, because
    every attribute was unquoted and HTML swallowed the trailing slash. The page
    rendered nothing but the polylines and still looked plausible. There is no
    reason to own that code when a plotting library does ticks, scales, legends
    and hover for free.
    """
    return dict(title=title, ylabel=ylabel, series=series, zero=zero,
                hlines=hlines, markers=markers, yerr=yerr)


def dashboard(specs, cols=2):
    """Every panel in one figure: shared hover, aligned axes, one legend-free
    grid. Series are named in the subplot titles instead of in a legend, so
    colours never have to be matched across panels."""
    specs = [s for s in specs if s]
    rows = -(-len(specs) // cols)
    titles = []
    for sp in specs:
        names = [f'<span style="color:{INK[i % len(INK)]}">{lab}</span>'
                 for i, (lab, *_ ) in enumerate(sp["series"]) if lab]
        names += [f'<span style="color:{INK[(i + 1) % len(INK)]}">{n}</span>'
                  for i, (n, _) in enumerate(sp["hlines"])]
        titles.append(f'<b>{sp["title"]}</b>'
                      + (f'  {" ".join(names)}' if names else "")
                      + f'  <span style="color:{AXIS}">· {sp["ylabel"]}</span>')
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        vertical_spacing=0.11, horizontal_spacing=0.07)
    for k, sp in enumerate(specs):
        r, c = k // cols + 1, k % cols + 1
        yerr = sp.get("yerr") or []
        for i, (label, x, y, smooth) in enumerate(sp["series"]):
            col = INK[i % len(INK)]
            if smooth:
                # Raw behind, trend in front: per-epoch values are jumpy because
                # every epoch trains on a fresh sample of a large buffer, so the
                # trend is the readable part and the scatter is its context.
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines", line=dict(color=col, width=1),
                    opacity=0.22, showlegend=False, hoverinfo="skip"), r, c)
                y = ema(y)
            err = yerr[i] if i < len(yerr) else None
            fig.add_trace(go.Scatter(
                x=x, y=y, name=label or sp["title"],
                mode="lines+markers" if sp["markers"] else "lines",
                line=dict(color=col, width=1.8), marker=dict(size=6),
                showlegend=False,
                error_y=(dict(type="data", array=err, color=col,
                              thickness=1, width=3) if err else None),
                hovertemplate=(f"{label or 'elo'}: %{{y:.0f}} ± "
                               f"%{{error_y.array:.0f}} (95% CI)<extra></extra>"
                               if err else
                               f"{label or sp['ylabel']}: %{{y:.4g}}<extra></extra>")),
                r, c)
        for i, (name, v) in enumerate(sp["hlines"]):
            fig.add_hline(y=v, row=r, col=c, line=dict(
                color=INK[(i + 1) % len(INK)], width=1, dash="dash"))
        if sp["zero"]:
            fig.update_yaxes(rangemode="tozero", row=r, col=c)
        else:
            # Percentile limits: one bad epoch -- a drain record, a warm-up
            # spike -- must not set the axis and flatten everything else.
            vals = []
            for i, (_, _, ys, _) in enumerate(sp["series"]):
                err = yerr[i] if i < len(yerr) else None
                for j, v in enumerate(ys):
                    if not math.isfinite(v):
                        continue
                    vals.append(v)
                    if err and j < len(err) and math.isfinite(err[j]):
                        vals.extend((v - err[j], v + err[j]))
            if vals:
                lo, hi = quantiles(vals, 0.01), quantiles(vals, 0.99)
                pad = (hi - lo) * 0.12 or abs(hi) * 0.1 or 1.0
                fig.update_yaxes(range=[lo - pad, hi + pad], row=r, col=c)
        if r == rows:
            fig.update_xaxes(title_text="minutes", row=r, col=c)

    fig.update_layout(
        height=330 * rows, margin=dict(l=48, r=14, t=34, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=AXIS, size=11,
                  family="ui-monospace,SFMono-Regular,Menlo,monospace"),
        hovermode="x unified",
        hoverlabel=dict(font_size=11, bgcolor="rgba(20,20,24,0.92)",
                        font_color="#eee", bordercolor=AXIS))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS,
                     ticks="outside", ticklen=3, title_standoff=6)
    fig.update_yaxes(gridcolor="rgba(138,138,138,0.20)", zeroline=False,
                     linecolor=AXIS, ticks="outside", ticklen=3)
    for a in fig.layout.annotations:          # subplot titles, left-aligned
        a.update(x=a.x - 0.5 / cols + 0.004, xanchor="left", font_size=12)
    return fig.to_html(include_plotlyjs=False, full_html=False,
                       default_height=f"{330 * rows}px",
                       config={"displaylogo": False, "responsive": True,
                               "modeBarButtonsToRemove":
                                   ["select2d", "lasso2d", "autoScale2d"]})


def quantiles(vals, q):
    """The q-quantile by nearest rank. No numpy: this module is imported by the
    trainer at the end of a run and should not need the scientific stack."""
    v = sorted(vals)
    return v[min(len(v) - 1, max(0, int(q * (len(v) - 1) + 0.5)))]


def checkpoints(run):
    """This run's players from a ladder shared by the whole experiment."""
    return [p for p in run["ladder"]["players"]
            if p["t"] is not None
            and (p.get("run") == run["path"]
                 or ("run" not in p and p["name"].startswith(run["name"] + ".")))]


def panels(runs):
    """Every panel of the dashboard, each overlaying every run."""
    mins = lambda r: [e["t"] / 60.0 for e in r["epochs"]]
    one = len(runs) == 1
    tag = lambda r: "" if one else r["name"]

    elo, elo_err, hl = [], [], []
    for r in runs:
        if not r["ladder"]:
            continue
        snaps = checkpoints(r)
        if snaps:
            elo.append((tag(r) or "snapshot", [p["t"] / 60.0 for p in snaps],
                        [p["elo"] for p in snaps], False))
            elo_err.append([ci95(p.get("se")) for p in snaps])
        p = next((q for q in r["ladder"]["players"] if q["name"] == "greedy"), None)
        if p and "greedy" not in [n for n, _ in hl]:
            hl.append(("greedy", p["elo"]))

    out = [panel("Strength vs training time", "elo (95% CI)", elo, hlines=hl,
                 markers=True, yerr=elo_err or None)]

    if one and "loss_old" in (runs[0]["epochs"] or [{}])[-1]:
        r = runs[0]
        out.append(panel("Value loss by row age", "huber", [
            ("old rows", mins(r), [e["loss_old"] for e in r["epochs"]], True),
            ("fresh rows", mins(r), [e["loss_new"] for e in r["epochs"]], True)]))
    else:
        out.append(panel("Value loss", "huber",
                         [(tag(r), mins(r), [e["loss"] for e in r["epochs"]], True)
                          for r in runs]))

    # Raw loss is not comparable across phases or runs, because the targets it
    # is measured against change scale: the greedy warm phase has a target std
    # near 0.5 and ReBeL settles nearer 0.2, so a falling loss can be nothing
    # but a shrinking target. Dividing by the target variance gives the share
    # of that variance the network fails to explain, which is scale-free and
    # is what says whether a phase actually fitted.
    unexplained = lambda e: e["loss"] / max(e["tgt_std"] ** 2, 1e-9)
    out.append(panel("Unexplained target variance", "loss / target variance",
                     [(tag(r), mins(r), [unexplained(e) for e in r["epochs"]], True)
                      for r in runs], zero=True))

    out.append(panel("Spread of predictions", "std",
                     [(tag(r) or "prediction", mins(r),
                       [e["probe_std"] for e in r["epochs"]], True) for r in runs]
                     + ([("target", mins(runs[0]),
                          [e["tgt_std"] for e in runs[0]["epochs"]], True)] if one else []),
                     zero=True))
    # Two throughput lines, and the gap between them is the information.
    #
    # Note what the trainer logs: `solves_per_s` is `rebel_solves / elapsed`,
    # both cumulative -- it is the *run average*, not the current rate. Plotting
    # it as "now" drew the same curve twice. The instantaneous rate has to come
    # from the differences between consecutive epochs, which is what this does.
    inst, cum = [], []
    for r in runs:
        eps = r["epochs"]
        if len(eps) < 2:
            continue
        nx, ny = [], []
        for a, b in zip(eps, eps[1:]):
            dt = b["t"] - a["t"]
            if dt > 0.5:
                nx.append(b["t"] / 60.0)
                ny.append(b["solves"] / dt)
        inst.append((f"{tag(r)} now".strip(), nx, ny, True))
        cum.append((f"{tag(r)} run average".strip(), mins(r),
                    [e["solves_per_s"] for e in eps], False))
    out.append(panel("Generation throughput", "solves/s", inst + cum, zero=True))
    if any("rows_per_s" in e for r in runs for e in r["epochs"]):
        out.append(panel("Replay generation throughput", "rows/s", [
            (tag(r), mins(r), [e.get("rows_per_s", 0) for e in r["epochs"]], False)
            for r in runs], zero=True))
    if any("effective_train_ratio" in e for r in runs for e in r["epochs"]):
        out.append(panel("Effective training ratio", "optimizer rows / solve", [
            (tag(r), mins(r), [e.get("effective_train_ratio", 0) for e in r["epochs"]], False)
            for r in runs], zero=True))
        out.append(panel("Passes per generated row", "optimizer rows / replay row", [
            (tag(r), mins(r), [e.get("train_row_ratio", 0) for e in r["epochs"]], False)
            for r in runs], zero=True))
        out.append(panel("Gradient clipping", "fraction of steps", [
            (tag(r), mins(r), [e.get("grad_clip_frac", 0) for e in r["epochs"]], True)
            for r in runs], zero=True))
        out.append(panel("Zero-sum residual", "max |v0 + v1|", [
            (tag(r), mins(r), [e.get("zero_sum_max", 0) for e in r["epochs"]], True)
            for r in runs], zero=True))
    out.append(panel("Replay age", "seconds retained", [
        (tag(r), mins(r), [e.get("buf_s", 0) for e in r["epochs"]], False)
        for r in runs], zero=True))
    # A refused build is the one decision a ReBeL agent answers without a
    # search; the micro-continuations that finish its coin play are searched.
    out.append(panel("Unsearched decisions", "fallbacks / decision", [
        (tag(r), mins(r), [e.get("node_caps", 0) / max(e.get("decisions", 1), 1)
                           for e in r["epochs"]], True)
        for r in runs], zero=True))
    # The horizon cuts a game at 256 coin plays and scores it a draw, and War
    # Chest has no draws. A rising rate means the ladder below is measuring a
    # game that is increasingly not the real one.
    out.append(panel("Games cut at horizon", "fraction",
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
             ("games cut at horizon", f"{last['horizon_frac']:.1%}")]
    warm = r.get("warm_end")
    if warm:
        cells.append(("warm fit left over",
                      f"{warm['loss'] / max(warm['tgt_std'] ** 2, 1e-9):.1%}"))
    if "effective_train_ratio" in last:
        cells += [("effective train ratio", f"{last['effective_train_ratio']:.3f} /solve"),
                  ("passes per row", f"{last.get('train_row_ratio', 0):.3f}"),
                  ("gradient clipped", f"{last['grad_clip_frac']:.1%}"),
                  ("max |v0+v1|", f"{last['zero_sum_max']:.2e}"),
                  ("replay age", f"{last['buf_s'] / 60:.1f} min")]
    return "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in cells)


def ladder_table(lad):
    if not lad:
        return "<p class=none>no ladder yet — <code>python train/ladder.py &lt;run&gt;</code></p>"
    players = [p for p in lad["players"] if p["name"] != "random"]
    pairs_in = [p for p in lad["pairs"] if "random" not in (p["a"], p["b"])]

    def prow(p):
        when = "—" if p["t"] is None else f"{p['t'] / 60:.0f} min"
        return (f"<tr><td>{p['name']}</td><td class=n>{when}</td>"
                f"<td class=n>{p['elo']:.0f}</td><td class=n>±{ci95(p.get('se')):.0f}</td>"
                f"<td class=n>{p['score']:.3f}</td></tr>")

    rows = "".join(prow(p) for p in sorted(players, key=lambda p: -p["elo"]))
    pairs = "".join(
        f"<tr><td>{p['a']} <span class=vs>vs</span> {p['b']}</td>"
        f"<td class=n>{p['w']}–{p['l']}–{p['d']}</td>"
        f"<td class=n>{p.get('n', p['w'] + p['l'] + p['d'])}</td>"
        f"<td class=n>{p['score']:.3f}</td></tr>"
        for p in sorted(pairs_in, key=lambda p: -(p.get("n", 0))))
    return (f"<div class=tw><table><thead><tr><th>player<th class=n>trained"
            f"<th class=n>elo<th class=n>95% CI<th class=n>score"
            f"</thead>{rows}</table></div>"
            f"<h3>Head to head</h3><div class=tw><table><thead><tr><th>pairing"
            f"<th class=n>W–L–D<th class=n>games<th class=n>score</thead>"
            f"{pairs}</table></div>")


CSS = """
:root{--bg:#fff;--ink:#111;--mut:#666;--line:#ddd;--grid:#eee}
@media (prefers-color-scheme:dark){
  :root{--bg:#111;--ink:#e8e8e8;--mut:#999;--line:#333;--grid:#262626}
}
:root[data-theme="dark"]{--bg:#111;--ink:#e8e8e8;--mut:#999;--line:#333;--grid:#262626}
:root[data-theme="light"]{--bg:#fff;--ink:#111;--mut:#666;--line:#ddd;--grid:#eee}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);
 font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 font-variant-numeric:tabular-nums}
h1{font-size:14px;font-weight:700;margin:0 0 2px}
h2{font-size:12px;font-weight:700;margin:22px 0 6px;color:var(--mut)}
h3{font-size:12px;font-weight:700;margin:16px 0 4px}
.meta{color:var(--mut);margin:0 0 6px}
.delta{display:flex;flex-wrap:wrap;gap:4px 16px;margin:0 0 10px;color:var(--mut)}
.delta span{color:var(--mut)}
.delta span.hit{color:var(--ink);font-weight:700}
.note{margin:0 0 8px;color:var(--ink)}
.js-plotly-plot{margin:0 0 4px}
dl{display:flex;flex-wrap:wrap;gap:0 22px;margin:4px 0}
dt{font-size:11px;color:var(--mut)}
dd{margin:0 0 4px}
table{border-collapse:collapse;margin:4px 0;font-size:12px}
th,td{text-align:left;padding:2px 14px 2px 0;white-space:nowrap}
th{color:var(--mut);font-weight:400;border-bottom:1px solid var(--line)}
td.n,th.n{text-align:right}
.tw{overflow-x:auto}
.none{color:var(--mut);font-size:11px}
footer{color:var(--mut);font-size:11px;margin-top:24px;border-top:1px solid var(--line);
 padding-top:8px}
"""

def page(runs, title, js):
    sha = runs[0]["cfg"].get("git", "")
    body = [f"<h1>{title}</h1>",
            '<p class="meta">'
            + (f"{sha} · " if sha and sha != "?" else "")
            + f"{len(runs[0]['epochs'])} ReBeL epochs</p>"]
    for r in runs:
        note = (r["cfg"] or {}).get("note") or ""
        if note:
            body.append(f'<p class="note">{"" if len(runs) == 1 else html.escape(r["name"]) + ": "}'
                        f"{html.escape(note)}</p>")
        body.append(f'<div class="delta">{cfg_html(r["cfg"])}</div>')
    body.append(dashboard(panels(runs)))
    for r in runs:
        body.append(f"<h3>{r['name']}</h3><dl>{health(r)}</dl>")
        body.append(ladder_table(r["ladder"]))
    body.append("<footer>Generated by train/report.py. Elo error bars are 95% "
                "confidence intervals (1.96 × Bradley-Terry standard error) on "
                "that player's rating, with greedy fixed at 0. Unexplained "
                "variance is loss / target-std².</footer>")
    return (f"<!doctype html><meta charset=utf-8><title>{title}</title>"
            f"{js}"
            f"<style>{CSS}</style>{''.join(body)}")


def cfg_html(cfg):
    return "".join(
        f'<span{" class=hit" if changed else ""}>{html.escape(k)}={html.escape(str(v))}</span>'
        for k, v, changed in config.knobs(cfg))


def write_all(runs_dir=RUNS_DIR):
    """Regenerate every run that left a log.json."""
    if not os.path.isdir(runs_dir):
        return
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name)
        if os.path.isfile(os.path.join(path, "log.json")):
            try:
                write([path])
            except Exception as e:
                print(f"[report] skip {path}: {e}", flush=True)


def write(runs, out=None, title=None, standalone=False):
    """Render run directories to one page. Plotly lives once in runs/.

    The Plotly bundle is written once into `runs/` and every report links to it
    by relative path, rather than being inlined into each page. Three megabytes
    per run adds up fast on a box that produces a report an hour, and one shared
    copy still travels with an `rsync` of the runs directory -- so the pages
    keep working with no network, which a CDN link would not.
    """
    loaded = [read(r) for r in runs]
    out = out or f"{runs[0]}/report.html"
    outdir = os.path.dirname(os.path.abspath(out))
    if standalone:
        # One file that renders anywhere, for sending to someone. Costs ~4.8 MB.
        js = f"<script>{plotly.offline.get_plotlyjs()}</script>"
    else:
        root = os.path.abspath(RUNS_DIR)
        path = os.path.join(root, JS)
        if not os.path.exists(path):
            os.makedirs(root, exist_ok=True)
            write_text(path, plotly.offline.get_plotlyjs())
            print(f"[report] wrote {path}", flush=True)
        js = f'<script src="{os.path.relpath(path, outdir)}" charset="utf-8"></script>'
    write_text(out, page(loaded, title or " · ".join(r["name"] for r in loaded), js))
    print(f"[report] {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*",
                    help="run directories (default: every runs/*/log.json)")
    ap.add_argument("-o", "--out", default=None,
                    help="output path (default: <first run>/report.html)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--standalone", action="store_true",
                    help="inline the plotting library (~4.8 MB) so the file "
                         "renders on its own, anywhere")
    args = ap.parse_args()
    if args.runs:
        write(args.runs, args.out, args.title, args.standalone)
    else:
        write_all()


if __name__ == "__main__":
    main()
