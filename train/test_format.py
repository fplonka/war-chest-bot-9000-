"""The format freeze's end-to-end test (pre-CUDA plan section 3).

Generate a few hundred rows through the real generation path, write a dump,
load it back through `Dump`, take a solve-aligned subset, make a batch, run a
few offline training steps, and evaluate a validation loss. Anything that
changes a shape -- the row layout, the expansion, the dedup key, the mirror,
the batch assembly -- breaks this test loudly instead of silently mis-training
later.

    python train/test_format.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import warchest
from value_net import Mlp
from train import Buffer, make_batch, value_loss
from dump import Dump
from offline import evaluate

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
ROW_BYTES = warchest.ROW_BYTES


def main():
    torch.manual_seed(3)
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    out = "/tmp/format_test"
    os.makedirs(out, exist_ok=True)

    # A small random net so generation produces real (noisy) targets; the
    # point is the plumbing, not the strength.
    net = Mlp(128, 32, 32, 16)
    net.push(0)

    print("[1/6] generating rows (random drafts, WP included)", flush=True)
    d = warchest.gen_data(16, 7, "rebel", depth=1, iters=8, explore=0.25,
                          random_draft=True)
    n = len(d["rows"]) // ROW_BYTES
    assert n > 200, f"expected a few hundred rows, got {n}"
    print(f"      {n} rows, {len(d['cc']) // CCOUNTS} configs, "
          f"{int(d['solves'])} solves, row_bytes={ROW_BYTES}", flush=True)

    print("[2/6] dumping through the real Buffer path", flush=True)
    buf = Buffer(200_000, 200_000 * 48)
    rows = np.asarray(d["rows"], np.uint8).reshape(-1, ROW_BYTES)
    cc = np.asarray(d["cc"], np.uint8).reshape(-1, CCOUNTS)
    cw = np.asarray(d["cw"], np.float32)
    cy = np.clip(np.asarray(d["cy"], np.float32), -1.0, 1.0)
    coff = np.asarray(d["coff"], np.int64)
    soff = np.asarray(d["soff"], np.int64)
    buf.add(rows, cc, cw.astype(np.float16), cy.astype(np.float16), coff, soff)
    dump_path = f"{out}/buffer.npz"
    got, gcc, gcp, gcw, gcy, gseg = buf.ordered()
    lo = buf.lo
    gsoff = np.concatenate([[0], buf.soff[(buf.soff > lo) & (buf.soff < buf.rows)] - lo,
                            [len(got)]])
    np.savez(dump_path, rows=got, cc=gcc, cp=gcp, cw=gcw, cy=gcy, seg=gseg,
             soff=gsoff, pubfeat=np.int32(PUBFEAT), cfeat=np.int32(CFEAT),
             ccounts=np.int32(CCOUNTS), cnorm=np.float32(CNORM),
             row_bytes=np.int32(ROW_BYTES), version=np.int32(warchest.ROW_FORMAT_VERSION),
             rules_hash=np.uint64(warchest.rules_table_hash()))

    print("[3/6] loading through Dump and checking the format pins", flush=True)
    dmp = Dump(dump_path)
    dmp.check(PUBFEAT, CCOUNTS)
    assert dmp.version == warchest.ROW_FORMAT_VERSION
    assert dmp.rules_hash == warchest.rules_table_hash()
    assert len(dmp.soff) >= 2 and dmp.soff[0] == 0 and dmp.soff[-1] == len(dmp)
    print(f"      {len(dmp)} rows, {len(dmp.soff) - 1} solve boundaries", flush=True)

    print("[4/6] solve-aligned split and batch assembly", flush=True)
    split = dmp.soff[-2]  # the newest solve block is the test set
    tr = dmp.rows(0, split)
    te = dmp.rows(split, len(dmp))
    rng = np.random.default_rng(0)
    b = make_batch(tr, rng, dev, augment=True)
    xpub, unit_ids, phi, inv, w, seg, y, nseg, ay = b
    assert xpub.shape == (len(tr[0]), PUBFEAT), xpub.shape
    assert unit_ids.shape == (len(tr[0]), warchest.NTYPE)
    assert phi.shape[1] == CFEAT
    assert seg.max() == 2 * len(tr[0]) - 1
    assert ay.shape == (len(tr[0]), warchest.AUX)
    assert torch.isfinite(xpub).all() and torch.isfinite(y).all()
    print(f"      batch {xpub.shape} unit_ids {unit_ids.shape} phi {phi.shape}", flush=True)

    print("[5/6] ten offline training steps", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(10):
        parts = make_batch(tr, rng, dev, augment=True)
        loss = value_loss(net, *parts[:-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(np.isfinite(losses)), losses
    print(f"      losses: {[round(x, 4) for x in losses]}", flush=True)

    print("[6/6] validation loss on the held-out solve block", flush=True)
    hl, hrms = evaluate(net, te, rng, dev)
    print(f"      test huber {hl:.6f} rms {hrms:.5f}", flush=True)
    assert np.isfinite(hl)
    print("format test OK", flush=True)


if __name__ == "__main__":
    main()
