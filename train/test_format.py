
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import warchest
from value_net import Net
from gpu_batch import make_batch
from train import Buffer, forward_values, losses

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
ROW_BYTES = warchest.ROW_BYTES


def empty_policy():
    return (np.zeros((0, warchest.ACT_BYTES), np.uint8),
            np.zeros(0, np.int64), np.zeros(0, np.int64),
            np.zeros(0, np.int64), np.zeros(0, np.float32),
            np.zeros(0, np.int64))


@torch.no_grad()
def evaluate(net, parts, rng, dev):
    batch = make_batch(parts, rng, dev)
    value = losses(net, *batch)
    rms = torch.sqrt(torch.mean((forward_values(net, batch) - batch[4]) ** 2))
    return float(value), float(rms)


def main():
    torch.manual_seed(3)
    torch.set_num_threads(4)
    dev = torch.device("cuda:0")
    net = Net().to(dev)
    net.push()

    print("[1/5] generating rows (random drafts, WP included)", flush=True)
    d = warchest.gen_data(4, 7, explore=0.25, random_draft=True)
    n = len(d["rows"]) // ROW_BYTES
    assert n > 200, f"expected a few hundred rows, got {n}"
    print(f"      {n} rows, {len(d['cc']) // CCOUNTS} configs, "
          f"{int(d['solves'])} solves, row_bytes={ROW_BYTES}", flush=True)

    print("[2/5] filling the real Buffer path", flush=True)
    buf = Buffer(200_000, 200_000 * 48)
    rows = np.asarray(d["rows"], np.uint8).reshape(-1, ROW_BYTES)
    cc = np.asarray(d["cc"], np.uint8).reshape(-1, CCOUNTS)
    cw = np.asarray(d["cw"], np.float32)
    cy = np.clip(np.asarray(d["cy"], np.float32), -1.0, 1.0)
    coff = np.asarray(d["coff"], np.int64)
    soff = np.asarray(d["soff"], np.int64)
    query = np.asarray(d["query"], np.uint8)
    source = np.where(query != 0, 2, 1).astype(np.uint8)
    truth = np.asarray(d["truth"], np.uint32).reshape(-1, 2)
    outcome = np.asarray(d["outcome"], np.float32).reshape(-1, 2)
    control = np.asarray(d["control"], np.uint8).reshape(len(rows), -1)
    created = np.asarray(d["created"], np.float64)
    td1 = np.asarray(d["td1"], np.uint8)
    replay = (rows, cc, cw.astype(np.float16), cy.astype(np.float16), coff,
              soff, source, truth, outcome, control, created, td1)
    buf.add(*replay)
    tiny = Buffer(max(n * 2, 8), max(n * 2, 8) * 48)
    for _ in range(8):
        tiny.add(*replay)
    assert tiny.soff.size < tiny.rows, (tiny.soff.size, tiny.rows)
    tiny.clear()
    assert tiny.soff.size == 0
    inner = buf.soff[(buf.soff > buf.lo) & (buf.soff < buf.rows)]
    assert inner.size, "a dump needs at least one interior solve boundary"
    print(f"      {len(buf)} rows, {inner.size + 1} solve boundaries", flush=True)

    print("[3/5] solve-aligned split and batch assembly", flush=True)
    split = int(inner[-1])
    block = lambda lo, hi: (lambda g: (*g[:6], empty_policy(), g[7]))(buf.gather(np.arange(lo, hi)))
    tr, te = block(buf.lo, split), block(split, buf.rows)
    rng = np.random.default_rng(0)
    b = make_batch(tr, rng, dev)
    xpub, phi, w, seg, y, nseg, policy, control = b
    assert control.shape == (len(tr[0]), warchest.N_HEXES), control.shape
    assert xpub.shape == (len(tr[0]), PUBFEAT), xpub.shape
    assert phi.shape[1] == CFEAT
    assert seg.max() == 2 * len(tr[0]) - 1
    assert nseg == 2 * len(tr[0])
    assert torch.allclose(torch.bincount(seg, w), torch.ones(nseg, device=w.device), atol=1e-6)
    assert not len(policy[0]) and not len(policy[1])
    assert torch.isfinite(xpub).all() and torch.isfinite(y).all()
    print(f"      batch {xpub.shape} phi {phi.shape}", flush=True)

    print("[4/5] ten offline training steps", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    seen = []
    for _ in range(10):
        parts = make_batch(tr, rng, dev)
        value = losses(net, *parts)
        opt.zero_grad(set_to_none=True)
        value.backward()
        opt.step()
        seen.append(float(value.detach()))
    assert all(np.isfinite(x) for x in seen), seen
    print("      value: " + " ".join(f"{v:.3f}" for v in seen), flush=True)

    print("[5/5] validation loss on the held-out solve block", flush=True)
    hl, hrms = evaluate(net, te, rng, dev)
    print(f"      test huber {hl:.6f} rms {hrms:.5f}", flush=True)
    assert np.isfinite(hl)
    print("format test OK", flush=True)


if __name__ == "__main__":
    main()
