"""How fast does the value function move, and therefore how stale may the buffer be?

The replay cap is set in rows, but what it buys is history, and what history
costs is staleness: a bootstrapped target was computed by whatever network
existed when the row was written, so a row of age `a` carries a label from a
function that has since moved. This measures how far it moves.

Take a run's snapshots, evaluate every one of them on one fixed set of
positions, and report the belief-weighted RMS difference between each pair
against the gap in wall clock. Call the time in which the network moves by its
own held-out error the **drift time** `tau`. A buffer spanning much more than
`tau` is training on labels from a different function; a buffer spanning much
less has thrown away data it could still have used.

    python train/drift.py runs/gpu_golden8 --set data/truth.npz

The set is only a source of positions and beliefs -- its targets are used for
nothing here, so any dump will do. The differences are affine-corrected as
well as raw: a network whose values have merely grown in scale has not really
moved, and a bootstrap tolerates a rescaling far better than it tolerates a
reordering. Measured on `gpu_golden8` the correction accounts for only 12-20%,
so the drift is real.

Reading it: with `R` rows per second from the run's log and `recent_mix`
drawing half the batch from the newest fifth, a cap of `C` rows spans `C / R`
and its batches have a mean age near `0.3 C / R`. Choosing `C` so that span is
around `2 tau` puts mean staleness at about `0.6 tau`.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump import Dump                              # noqa: E402
from export_weights import load as load_checkpoint  # noqa: E402
from offline import subset                          # noqa: E402
from train import make_batch                        # noqa: E402


def predictions(path, parts, dev, rng):
    """Every config's value under one checkpoint, and the beliefs."""
    net = load_checkpoint(path).to(dev).eval()
    v, w = [], []
    for i in range(0, len(parts[0]), 4096):
        ids = np.arange(i, min(i + 4096, len(parts[0])))
        b = make_batch(subset(parts, ids), rng, dev, False)
        with torch.no_grad():
            v.append(net(b[0], b[1], b[2], b[3], b[4], b[5], b[7]).numpy())
        w.append(b[4].numpy())
    return np.concatenate(v), np.concatenate(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="a run directory with log.json and snapshots")
    ap.add_argument("--set", default="data/truth.npz", help="positions and beliefs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    d = Dump(args.set)
    parts = d.rows(0, len(d))
    dev, rng = torch.device(args.device), np.random.default_rng(0)
    with open(f"{args.run}/log.json") as f:
        log = json.load(f)
    snaps = [(s["label"], s["t"], f"{args.run}/{s['file']}") for s in log["snapshots"]]
    if len(snaps) < 2:
        raise SystemExit(f"{args.run}: needs at least two snapshots")

    pred = {}
    for label, _, path in snaps:
        pred[label], w = predictions(path, parts, dev, rng)
    wsum = w.sum()
    wrms = lambda x: math.sqrt(float((w * x ** 2).sum() / wsum))
    wmean = lambda x: float((w * x).sum() / wsum)

    print(f"[drift] {args.run}: {len(d)} positions, {len(w)} configs, "
          f"set from {args.set}", flush=True)
    print(f"\n{'gap/min':>8} {'pair':>18} {'sd':>7} {'drift':>7} {'slope':>6} "
          f"{'affine':>7}")
    # The first snapshot is the warm network, a different animal; pairs within
    # the ReBeL phase are the ones that describe the loop.
    for i in range(len(snaps)):
        for j in range(i + 1, len(snaps)):
            (a, ta, _), (b, tb, _) = snaps[i], snaps[j]
            x, y = pred[a], pred[b]
            mx, my = wmean(x), wmean(y)
            slope = float((w * (x - mx) * (y - my)).sum() / (w * (x - mx) ** 2).sum())
            print(f"{(tb - ta) / 60:8.0f} {a + ' -> ' + b:>18} {wrms(y - my):7.4f} "
                  f"{wrms(y - x):7.4f} {slope:6.2f} "
                  f"{wrms(y - (my + slope * (x - mx))):7.4f}")
    print("\nsd is the later network's own spread; drift is the raw RMS change;\n"
          "affine is what survives the best rescaling of the earlier network.\n"
          "Compare drift against the network's held-out error: the gap where\n"
          "they are equal is tau, and the cap worth running is about 2 tau of\n"
          "generation (`rows/s` from the run's log, or `buf_s` in its epochs).")


if __name__ == "__main__":
    main()
