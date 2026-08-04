"""Show how a training run is going. Opens a window.

    python train/plot.py runs/hour01

Reads `log.json`, which the trainer rewrites at every gate, so this works on a
run that is still going.

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


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/latest"
    try:
        with open(f"{out}/log.json") as f:
            log = json.load(f)
    except FileNotFoundError:
        sys.exit(f"{out}/log.json not found — is the run started, and is the path right?")
    reb = [r for r in log["epochs"] if r["phase"] == "rebel"]
    if not reb:
        sys.exit(f"{out}: no ReBeL epochs yet — still warming up")
    t = np.array([r["t"] for r in reb]) / 60.0
    gate = log.get("gate", [])

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")

    # 1. Value loss. Linear: it settles inside a factor of about two, where a
    #    log axis costs readable ticks and shows nothing extra.
    ax = axes[0][0]
    series(ax, t, [r["loss"] for r in reb], BLUE, "loss")
    style(ax, "Value loss", "huber")

    # 2. Strength against the reference opponents. A win rate, so it gets a
    #    0.5 line, and promotions are marked where the champion was replaced.
    ax = axes[0][1]
    if gate:
        gt = np.array([g["t"] for g in gate]) / 60.0
        for key, color in (("champ", BLUE), ("greedy", GREEN), ("init", MAGENTA)):
            if key in gate[0]:
                v = [g[key] for g in gate]
                ax.plot(gt, v, "-o", color=color, linewidth=2, markersize=6, label=key)
                ax.annotate(f" {key}", (gt[-1], v[-1]), color=color, fontsize=8,
                            va="center")
        promo = [g["t"] / 60.0 for g in gate if g.get("promoted")]
        for p in promo:
            ax.axvline(p, color=MUTED, linewidth=0.8, linestyle=":", zorder=0)
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--", zorder=0)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, frameon=False, loc="lower right", ncols=3)
        n = sum(1 for g in gate if g.get("promoted"))
        ax.set_title(f"Strength vs reference  ·  {n} promotions"
                     f"  (dotted = promoted, dashed = even)",
                     fontsize=10, color=INK, loc="left", pad=8)
    else:
        ax.text(0.5, 0.5, "no gate yet", ha="center", va="center",
                color=MUTED, fontsize=9, transform=ax.transAxes)
        ax.set_title("Strength vs reference", fontsize=10, color=INK, loc="left",
                     pad=8)
    style(ax, ax.get_title(), "win rate")

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

    for ax in axes[1]:
        ax.set_xlabel("minutes", fontsize=8, color=MUTED)

    buf = reb[-1]["buf"]
    fig.suptitle(f"{out}  ·  {len(reb)} ReBeL epochs  ·  {t[-1]:.0f} min"
                 f"  ·  buffer {buf:,}",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.94, 0.95))
    plt.show()


if __name__ == "__main__":
    main()
