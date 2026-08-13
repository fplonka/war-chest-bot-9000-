"""Show how a training run went. Opens a window.

    python train/plot.py runs/hour01

Reads `log.json`, which the trainer rewrites every epoch, so this works on a run
that is still going, and `ladder.json` if `ladder.py` has already rated the
snapshots.

Four panels sharing one time axis rather than one crowded chart with two y-axes:
the measures have unrelated scales, and a second y-axis invites reading a
crossing point that means nothing.

Faint dots are the raw per-epoch values; the solid line is an exponential
moving average. Per-epoch training loss is genuinely jumpy — each epoch trains
on a fresh sample from a large buffer — so the trend is the readable part and
the scatter is the honest context for it.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-3 of the reference palette, in their validated order.
BLUE, GREEN, MAGENTA = "#2a78d6", "#008300", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e2"


def ema(y, span=12):
    """Exponential moving average, the standard smoother for training curves."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    a = 2.0 / (span + 1.0)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = a * y[i] + (1 - a) * out[i - 1]
    return out


def series(ax, x, y, color, label, span=12, direct=True):
    """Raw scatter behind a smoothed line.

    Direct-labelled at its right end when the series are far enough apart to
    read; where they nearly coincide the labels would collide, so those panels
    pass direct=False and carry a legend instead.
    """
    if len(x) == 0:
        return
    ax.plot(x, y, ".", color=color, alpha=0.18, markersize=3, zorder=1)
    s = ema(y, span)
    ax.plot(x, s, "-", color=color, linewidth=2, zorder=2, label=label)
    if direct:
        ax.annotate(f"  {label}", (x[-1], s[-1]), color=color, fontsize=8,
                    va="center")


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def elo_panel(ax, ladder):
    """Elo against minutes trained: the run's headline.

    Snapshots carry a training time and form the curve; Greedy and Random have
    no training time at all, so they are horizontal reference lines rather than
    points on it. The band is one standard error, which on a few hundred games
    is wide enough that it should be visible whenever the curve is read.
    """
    if not ladder:
        ax.text(0.5, 0.5, "no ladder yet  (python train/ladder.py <run>)",
                ha="center", va="center", color=MUTED, fontsize=9,
                transform=ax.transAxes)
        style(ax, "Elo vs training time", "elo")
        return
    snaps = [p for p in ladder["players"] if p["t"] is not None]
    x = np.array([p["t"] for p in snaps]) / 60.0
    y = np.array([p["elo"] for p in snaps])
    se = np.array([p["se"] for p in snaps])
    ax.fill_between(x, y - se, y + se, color=BLUE, alpha=0.15, zorder=1)
    ax.plot(x, y, "-o", color=BLUE, linewidth=2, markersize=6, zorder=3,
            label="snapshot")
    # Direct labels in the right margin, so nothing sits on top of the marks.
    at_right = ax.get_yaxis_transform()
    for p, color in (("greedy", GREEN), ("random", MAGENTA)):
        ref = next((q for q in ladder["players"] if q["name"] == p), None)
        if ref is None:
            continue
        ax.axhline(ref["elo"], color=color, linewidth=1.4, linestyle="--", zorder=2)
        ax.annotate(f" {p}", (1.0, ref["elo"]), xycoords=at_right, color=color,
                    fontsize=8, va="center", annotation_clip=False)
    ax.annotate(" final", (1.0, y[-1]), xycoords=at_right, color=BLUE, fontsize=8,
                va="center", annotation_clip=False)
    style(ax, f"Elo vs training time  ·  {ladder['games_per_pair']} games per pairing"
              f"  (band = 1 s.e.)", "elo")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/latest"
    try:
        with open(f"{out}/log.json") as f:
            log = json.load(f)
    except FileNotFoundError:
        sys.exit(f"{out}/log.json not found — is the run started, and is the path right?")
    try:
        with open(f"{out}/ladder.json") as f:
            ladder = json.load(f)
    except FileNotFoundError:
        ladder = None
    reb = [r for r in log["epochs"] if r["phase"] == "rebel"]
    if not reb:
        sys.exit(f"{out}: no ReBeL epochs yet — still warming up")
    t = np.array([r["t"] for r in reb]) / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")

    # 1. Strength, in Elo, against how long the network had been trained when
    #    each snapshot was taken. The only strength measurement the run makes.
    elo_panel(axes[0][0], ladder)

    # 2. Value loss, plus the age buckets: bootstrapped targets are written
    #    by past versions of the net, so old rows carry stale labels. If the
    #    old-row loss falls while the fresh-row loss rises, training is
    #    overfitting the buffer. Linear: it settles inside a factor of about
    #    two, where a log axis costs readable ticks and shows nothing extra.
    ax = axes[0][1]
    series(ax, t, [r["loss"] for r in reb], BLUE, "loss")
    if "loss_old" in reb[-1]:
        series(ax, t, [r["loss_old"] for r in reb], MAGENTA, "old rows", direct=False)
        series(ax, t, [r["loss_new"] for r in reb], GREEN, "fresh rows", direct=False)
        ax.legend(fontsize=8, frameon=False, loc="upper right", ncols=3)
    style(ax, "Value loss  (old vs fresh rows = staleness)", "huber")

    # 3. The degeneracy canary. If the spread of the network's predictions
    #    collapses toward zero, the value function has gone flat -- the failure
    #    a falling loss curve hides.
    ax = axes[1][0]
    series(ax, t, [r["tgt_std"] for r in reb], BLUE, "target spread", direct=False)
    series(ax, t, [r["probe_std"] for r in reb], GREEN, "prediction spread", direct=False)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncols=2)
    style(ax, "Spread of values  (collapse toward 0 = degenerate)", "std")

    # 4. Throughput, as decisions solved per second.
    ax = axes[1][1]
    dps = [r["decisions"] / max(r["gen_s"], 1e-6) for r in reb]
    series(ax, t, dps, BLUE, "decisions/s")
    ax.set_ylim(bottom=0)
    style(ax, "Generation throughput", "per second")

    for ax in axes.ravel():
        ax.set_xlabel("minutes", fontsize=8, color=MUTED)

    cfg = log.get("cfg", {})
    buf = reb[-1]["buf"]
    fig.suptitle(f"{out}  ·  {len(reb)} ReBeL epochs  ·  {t[-1]:.0f} min"
                 f"  ·  depth {cfg.get('depth', '?')}, {cfg.get('iters', '?')} CFR iters"
                 f"  ·  buffer {buf:,}",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.95, 0.95))
    if len(sys.argv) > 2:
        fig.savefig(sys.argv[2], dpi=150)
        print(f"wrote {sys.argv[2]}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
