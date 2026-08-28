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
import mirror
from value_net import Net
from train import Buffer, forward_values, losses, make_batch
from dump import Dump

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
    dev = torch.device("cpu")
    out = "/tmp/format_test"
    os.makedirs(out, exist_ok=True)

    # The point is the replay plumbing, not search strength.
    net = Net()
    net.push()

    print("[1/6] generating rows (random drafts, WP included)", flush=True)
    d = warchest.gen_data(1, 7, explore=0.25, random_draft=True)
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
    query = np.asarray(d["query"], np.uint8)
    source = np.where(query != 0, 2, 1).astype(np.uint8)
    truth = np.asarray(d["truth"], np.uint32).reshape(-1, 2)
    outcome = np.asarray(d["outcome"], np.float32).reshape(-1, 2)
    created = np.asarray(d["created"], np.float64)
    td1 = np.asarray(d["td1"], np.uint8)
    replay = (rows, cc, cw.astype(np.float16), cy.astype(np.float16), coff,
              soff, source, truth, outcome, created, td1)
    buf.add(*replay)
    tiny = Buffer(max(n * 2, 8), max(n * 2, 8) * 48)
    for _ in range(8):
        tiny.add(*replay)
    assert tiny.soff.size < tiny.rows, (tiny.soff.size, tiny.rows)
    tiny.clear()
    assert tiny.soff.size == 0
    dump_path = f"{out}/buffer.npz"
    got, gcc, gcp, gcw, gcy, gseg, _ = buf.ordered()
    lo = buf.lo
    gsoff = np.concatenate([[0], buf.soff[(buf.soff > lo) & (buf.soff < buf.rows)] - lo,
                            [len(got)]])
    np.savez(dump_path, rows=got, cc=gcc, cp=gcp, cw=gcw, cy=gcy, seg=gseg,
             soff=gsoff, pubfeat=np.int32(PUBFEAT), cfeat=np.int32(CFEAT),
             ccounts=np.int32(CCOUNTS), cnorm=np.float32(CNORM),
             row_bytes=np.int32(ROW_BYTES),
             rules_hash=np.uint64(warchest.rules_table_hash()))

    print("[3/6] loading through Dump and checking the format pins", flush=True)
    dmp = Dump(dump_path)
    dmp.check(PUBFEAT, CCOUNTS)
    assert dmp.rules_hash == warchest.rules_table_hash()
    assert len(dmp.soff) >= 2 and dmp.soff[0] == 0 and dmp.soff[-1] == len(dmp)
    print(f"      {len(dmp)} rows, {len(dmp.soff) - 1} solve boundaries", flush=True)

    print("[4/6] solve-aligned split and batch assembly", flush=True)
    split = dmp.soff[-2]  # the newest solve block is the test set
    tr = (*dmp.rows(0, split), empty_policy())
    te = (*dmp.rows(split, len(dmp)), empty_policy())
    rng = np.random.default_rng(0)
    b = make_batch(tr, rng, dev)
    xpub, phi, w, seg, y, nseg, policy = b
    assert xpub.shape == (len(tr[0]), PUBFEAT), xpub.shape
    assert phi.shape[1] == CFEAT
    assert seg.max() == 2 * len(tr[0]) - 1
    assert nseg == 2 * len(tr[0])
    assert torch.allclose(torch.bincount(seg, w), torch.ones(nseg), atol=1e-6)
    assert not len(policy[0]) and not len(policy[1])
    assert torch.isfinite(xpub).all() and torch.isfinite(y).all()
    trows, tcc, tcp = tr[0], tr[1], tr[2]
    mirror.self_check_rows(trows)
    mirror.self_check(xpub.numpy())
    print(f"      batch {xpub.shape} phi {phi.shape}", flush=True)

    print("[5/6] ten offline training steps", flush=True)
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

    print("[6/6] validation loss on the held-out solve block", flush=True)
    hl, hrms = evaluate(net, te, rng, dev)
    print(f"      test huber {hl:.6f} rms {hrms:.5f}", flush=True)
    assert np.isfinite(hl)
    print(f"row mirror matches State::mirror on {mirror.check_against_engine()} states",
          flush=True)
    print("format test OK", flush=True)


if __name__ == "__main__":
    main()
