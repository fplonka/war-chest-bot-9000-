"""Freeze one replay buffer to disk, so architecture questions stop needing runs.

A ReBeL value target is a deterministic function of the row it is stored
against -- the CFR root value of the subgame at `(state, ctx, beliefs)` --
so a dump is a noise-free supervised dataset. Two architectures fitted to the
same dump can be compared in minutes, where two self-play runs disagree by more
than the effect being measured and take an hour each.

    python tools/makedump.py --out /workspace/dump.npz --solves 40000 \\
        --weights runs/cohorts10/snap_02.pt

The weights matter: the targets are what *that* network's search produced, so a
dump is a snapshot of one point in training, not of the game.
"""

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "train")
import warchest  # noqa: E402
from export_weights import load as load_checkpoint  # noqa: E402
from train import Buffer, ingest  # noqa: E402
from value_net import Net  # noqa: E402

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
ROW_BYTES = warchest.ROW_BYTES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workspace/dump.npz")
    p.add_argument("--solves", type=int, default=40000)
    p.add_argument("--weights", required=True)
    p.add_argument("--roots", default="")
    p.add_argument("--devices", default="0,1")
    p.add_argument("--threads", type=int, default=36)
    p.add_argument("--s", type=int, default=512)
    p.add_argument("--c", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    net = Net()
    net.load_state_dict(load_checkpoint(args.weights).state_dict())
    net.push()

    farm = warchest.SolveFarm(
        args.seed, args.threads,
        s=args.s, c=args.c, cfr="sog",
        recursive_rate=0.1, devices=[int(d) for d in args.devices.split(",")],
        roots=args.roots or None,
    )
    buf = Buffer(args.solves * 2, args.solves * 2 * 48, torch.device("cpu"))
    start, got = time.time(), 0
    while got < args.solves:
        d = farm.collect(solves=min(4096, args.solves - got))
        ingest(buf, d)
        got += int(d["solves"])
        print(f"  {got}/{args.solves} solves, {got / (time.time() - start):.1f}/s",
              flush=True)

    x, cc, cp, cw, cy, seg, _ = buf.ordered()
    to_numpy = lambda value: value.cpu().numpy() if torch.is_tensor(value) else value
    x, cc, cp, cw, cy, seg = map(to_numpy, (x, cc, cp, cw, cy, seg))
    lo = buf.lo
    soff = np.concatenate([[0],
                           buf.soff[(buf.soff > lo) & (buf.soff < buf.rows)] - lo,
                           [len(x)]])
    np.savez(args.out, rows=x, cc=cc, cp=cp, cw=cw, cy=cy, seg=seg, soff=soff,
             pubfeat=np.int32(PUBFEAT), cfeat=np.int32(CFEAT),
             ccounts=np.int32(CCOUNTS), cnorm=np.float32(CNORM),
             row_bytes=np.int32(ROW_BYTES),
             rules_hash=np.uint64(warchest.rules_table_hash()))
    print(f"wrote {args.out}: {len(x)} rows, {len(soff) - 1} solves")


if __name__ == "__main__":
    main()
