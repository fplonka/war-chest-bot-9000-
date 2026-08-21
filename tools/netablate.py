"""What each part of the value network costs, and what it is worth.

Two thirds of a solve's arithmetic is the trunk and the rest is the join, and
neither has ever been sized against anything. This fits several architectures to
*one frozen dump* and reports held-out value error beside the arithmetic each
would cost a solve. A dump is a noise-free supervised dataset, so the comparison
takes minutes and does not need a self-play run -- which is the only reason a
question like "are eight trunk blocks worth their 58%" is answerable at all.

    python tools/netablate.py --dump /workspace/dump.npz --steps 3000

The split is by solve and by recency: rows from one epoch come from the same
handful of games, so a random split leaks its answers into the training set.
"""

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "train")
import mirror  # noqa: E402
import value_net  # noqa: E402
import warchest  # noqa: E402
from dump import Dump  # noqa: E402
from train import expand_batch, public_sizes  # noqa: E402

CNORM = warchest.CNORM
ROW_BYTES = warchest.ROW_BYTES

# `(name, BLOCKS, C, JBLOCKS, JW, D, POOL, CFGH)`. The first is production.
VARIANTS = [
    ("baseline",     8, 96, 3, 128, 256, 64, 128),
    ("trunk 6",      6, 96, 3, 128, 256, 64, 128),
    ("trunk 4",      4, 96, 3, 128, 256, 64, 128),
    ("trunk 2",      2, 96, 3, 128, 256, 64, 128),
    ("C 64",         8, 64, 3, 128, 256, 64, 128),
    ("C 64, 4",      4, 64, 3, 128, 256, 64, 128),
    ("join 1",       8, 96, 1, 128, 256, 64, 128),
    ("join 64w",     8, 96, 3,  64, 256, 64, 128),
    ("join 1, 64w",  8, 96, 1,  64, 256, 64, 128),
    ("D 128",        8, 96, 3, 128, 128, 64, 128),
    ("small",        4, 64, 1,  64, 128, 64,  96),
]


def flops(blocks, c, jblocks, jw, d, pool, cfgh):
    """Multiply-accumulates a solve would spend, at the production budget.

    The shape is measured, not assumed: a solve at SoG(512, 8) builds 8,373
    network rows and its iterations sum to 258,690 rows, with 5.05 configs a
    query. The trunk and the board's join seed run once a row; the join and the
    readout run twice a row every iteration.
    """
    rows, row_iters, cfgs = 8373.0, 258690.0, 5.05
    nhex, loose = 37.0, 15.0
    once = blocks * (113.0 * c * c) + (2 * c + loose) * d + d * jw
    per_query = (2 * pool + 1) * jw + jblocks * jw * jw + jw * d + cfgs * d
    return 2e-9 * (rows * once + 2 * row_iters * per_query)


def build(dev, blocks, c, jblocks, jw, d, pool, cfgh):
    value_net.BLOCKS, value_net.C = blocks, c
    value_net.JBLOCKS, value_net.JW = jblocks, jw
    value_net.D, value_net.POOL, value_net.CFGH = d, pool, cfgh
    value_net.JOIN_IN = 2 * pool + 1
    return value_net.Net().to(dev)


def batch(dump, lo, hi, dev):
    rows, cc, cp, cw, cy, seg = dump.rows(lo, hi)
    n = len(rows)
    hand, fd, bag = public_sizes(cc, cp, seg, n)
    views = np.empty((2 * n, ROW_BYTES), np.uint8)
    views[0::2] = rows
    views[1::2] = mirror.mirror_rows(rows)
    sizes = []
    for a in (hand, fd, bag):
        pair = np.empty((2 * n, 2), np.uint8)
        pair[0::2] = a
        pair[1::2] = a[:, ::-1]
        sizes.append(pair)
    x = expand_batch(views, *sizes)
    t = lambda a, k=torch.float32: torch.as_tensor(a, dtype=k, device=dev)
    return (t(x), t(cc.astype(np.float32) / CNORM), t(cw),
            t(seg, torch.long), t(cy), 2 * n)


def value_loss(net, b):
    v = net(b[0], b[1], b[2], b[3], b[5])
    per = torch.nn.functional.smooth_l1_loss(v, b[4], reduction="none", beta=0.5)
    total = torch.zeros(b[5], dtype=per.dtype, device=per.device)
    count = torch.zeros(b[5], dtype=per.dtype, device=per.device)
    total.index_add_(0, b[3], per)
    count.index_add_(0, b[3], torch.ones_like(per))
    return (total / count.clamp(min=1)).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", default="/workspace/dump.npz")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    dev = torch.device(args.device)
    d = Dump(args.dump)
    d.check(warchest.PUBFEAT, warchest.CCOUNTS)
    # Held out by recency at a solve boundary: the last tenth of the rows.
    cut = int(d.soff[np.searchsorted(d.soff, int(0.9 * len(d))) - 1])
    print(f"{len(d)} rows, train [0,{cut}) test [{cut},{len(d)})")
    tests = [batch(d, i, min(i + args.batch, len(d)), dev)
             for i in range(cut, len(d) - args.batch, args.batch)][:16]
    var = float(np.var(d.cy.astype(np.float32)))

    print(f"\ntarget variance {var:.5f}\n")
    print(f"{'variant':>14} {'GFLOP/solve':>12} {'rel':>6} {'params':>9} "
          f"{'test L':>9} {'L/var':>7} {'fit s':>7}")
    base = None
    for name, *shape in VARIANTS:
        torch.manual_seed(7)
        net = build(dev, *shape)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        rng = np.random.default_rng(3)
        t0 = time.time()
        for _ in range(args.steps):
            lo = int(rng.integers(0, max(cut - args.batch, 1)))
            b = batch(d, lo, lo + args.batch, dev)
            opt.zero_grad(set_to_none=True)
            value_loss(net, b).backward()
            opt.step()
        with torch.no_grad():
            test = float(np.mean([float(value_loss(net, b)) for b in tests]))
        g = flops(*shape)
        base = base or g
        print(f"{name:>14} {g:>12.1f} {g / base:>6.2f} "
              f"{sum(q.numel() for q in net.parameters()):>9d} "
              f"{test:>9.5f} {test / var:>7.3f} {time.time() - t0:>7.0f}")


if __name__ == "__main__":
    main()
