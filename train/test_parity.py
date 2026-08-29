
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import warchest
from value_net import Net

PUBFEAT = warchest.PUBFEAT
N_HEXES = warchest.N_HEXES
HEX_CH = warchest.HEX_CH
HEX_FACTS = warchest.HEX_FACTS
NTYPE = warchest.NTYPE
NSLOT = warchest.NSLOT
PILE_COUNTS = warchest.PILE_COUNTS
CARD_FEATS = warchest.CARD_FEATS
CCOUNTS = warchest.CCOUNTS
N_KINDS = warchest.N_KINDS
CNORM = warchest.CNORM
OFF_PILES = warchest.OFF_PILES
OFF_CARDS = warchest.OFF_CARDS
OFF_LOOSE = warchest.OFF_LOOSE


def random_net(seed):
    torch.manual_seed(seed)
    net = Net()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.normal_(0, 0.08)
        for module in net.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight.normal_(1, 0.2)
                module.bias.normal_(0, 0.1)
    return net


def public_rows(rng, rows):
    x = rng.standard_normal((rows, PUBFEAT)).astype(np.float32)
    hexes = x[:, :N_HEXES * HEX_CH].reshape(rows, N_HEXES, HEX_CH)
    hexes[:, :, HEX_FACTS:] = 0
    occupant = rng.integers(0, NTYPE + 1, (rows, N_HEXES))
    r, h = np.nonzero(occupant < NTYPE)
    hexes[r, h, HEX_FACTS + occupant[r, h]] = 1
    return x


def belief(rng, sizes, zero_counts=False):
    sizes = np.asarray(sizes, np.int64)
    seg = np.repeat(np.arange(len(sizes), dtype=np.uint32), sizes)
    n = int(sizes.sum())
    phi = (np.zeros((n, CCOUNTS), np.float32) if zero_counts else
           rng.integers(0, 6, (n, CCOUNTS)).astype(np.float32) / CNORM)
    w, at = np.empty(n, np.float32), 0
    for k in sizes:
        if k:
            v = rng.random(int(k)).astype(np.float32)
            w[at:at + k] = v / v.sum()
            at += int(k)
    return seg, phi, w


def run(net, xpub, phi, weight, seg, queries):
    with torch.no_grad():
        v = net(torch.from_numpy(np.ascontiguousarray(xpub)),
                torch.from_numpy(np.ascontiguousarray(phi)),
                torch.from_numpy(weight),
                torch.from_numpy(seg.astype(np.int64)), queries)
    return v.numpy()


def slot_permutation(perm):
    full = np.concatenate([perm, NSLOT + perm])
    idx = np.arange(PUBFEAT)
    hexes = idx[:N_HEXES * HEX_CH].reshape(N_HEXES, HEX_CH)
    hexes[:, HEX_FACTS:] = hexes[:, HEX_FACTS:][:, full]
    piles = idx[OFF_PILES:OFF_CARDS].reshape(NTYPE, PILE_COUNTS)
    piles[:] = piles[full]
    cards = idx[OFF_CARDS:OFF_LOOSE].reshape(NTYPE, CARD_FEATS)
    cards[:] = cards[full]
    return idx


def slot_invariance(net, rng, perms=6):
    sizes = [4, 3, 6, 2, 5, 5]
    queries = len(sizes)
    xpub = public_rows(rng, queries // 2)
    seg, phi, weight = belief(rng, sizes)
    base = run(net, xpub, phi, weight, seg, queries)
    spread = float(base.std())
    assert spread > 1e-3, "degenerate invariance inputs"
    worst = 0.0
    for _ in range(perms):
        perm = rng.permutation(NSLOT)
        px = xpub[:, slot_permutation(perm)]
        pphi = phi.reshape(-1, 3, NSLOT)[:, :, perm].reshape(-1, CCOUNTS)
        got = run(net, px, pphi, weight, seg, queries)
        dv = float(np.max(np.abs(got - base)))
        worst = max(worst, dv)
        print(f"  slots {perm}  values {dv:.2e}")
    tol = 1e-4 * max(1.0, float(np.abs(base).max()))
    assert worst < tol, (f"slot permutation is not a relabelling: {worst:.3e} "
                         f"against a value spread of {spread:.3e}")
    print(f"slot invariance ok: worst {worst:.3e}, {worst / spread:.1e} of the "
          f"value spread ({spread:.3e})")

def offboard_pile_visibility(net, rng):
    sizes = [1, 1]
    xpub = public_rows(rng, len(sizes) // 2)
    hexes = xpub[:, :N_HEXES * HEX_CH].reshape(len(sizes) // 2, N_HEXES, HEX_CH)
    hexes[:, :, HEX_FACTS] = 0.0
    seg, phi, weight = belief(rng, sizes)
    base = run(net, xpub, phi, weight, seg, len(sizes))
    changed = xpub.copy()
    pile = OFF_PILES
    changed[0, pile:pile + PILE_COUNTS] += 2.0
    got = run(net, changed, phi, weight, seg, len(sizes))
    movement = float(np.max(np.abs(got - base)))
    assert movement > 1e-5, "an off-board unit's piles are invisible"
    print(f"off-board pile visibility ok: movement {movement:.3e}")


def packed_row_cuda_parity():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requires a working GPU")
    rows = np.frombuffer(bytes(warchest.sample_rows(128, 19)), np.uint8)
    rows = rows.reshape(-1, warchest.ROW_BYTES)[:2048]
    mirrored = np.frombuffer(bytes(warchest.mirror_rows(rows.ravel())), np.uint8)
    rows = np.concatenate([rows, mirrored.reshape(rows.shape)]).copy()
    assert len(rows) >= 2048, f"only {len(rows)} real rows"
    want = np.asarray(warchest.expand_rows(rows.ravel()), np.float32)
    want = want.reshape(len(rows), warchest.PUBFEAT)
    device = torch.device("cuda:0")
    packed = torch.as_tensor(np.ascontiguousarray(rows), device=device)
    cards = torch.as_tensor(
        np.asarray(warchest.card_features_table(), np.float32), device=device)
    locations = torch.as_tensor(
        np.asarray(warchest.hex_location_flags(), np.uint8), device=device)
    got = torch.empty_like(torch.as_tensor(want, device=device))
    stream = torch.cuda.current_stream(device)
    warchest.expand_rows_cuda(
        packed.data_ptr(), cards.data_ptr(), locations.data_ptr(), got.data_ptr(),
        len(rows), stream.cuda_stream, 0)
    got = got.cpu().numpy()
    exact = (want == 0.0) | (want == 1.0)
    assert np.array_equal(got[exact], want[exact]), "one-hot expansion drift"
    np.testing.assert_allclose(got[~exact], want[~exact], rtol=0, atol=1e-6)
    print(f"packed-row CUDA parity ok: {len(rows)} real rows, mirrors included")


def main():
    rng = np.random.default_rng(11)
    net = random_net(7)
    net.push()
    slot_invariance(net, rng)
    offboard_pile_visibility(net, rng)
    packed_row_cuda_parity()


if __name__ == "__main__":
    main()
