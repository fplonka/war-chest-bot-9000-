"""Parity between the PyTorch and Rust v5 forwards, and the slot invariance
both of them owe the draft.

Two tests, both on random weights so that neither can pass by accident:

* **Blob parity.** `Net.flat()` writes the flat weight blob and `V5Layout::new`
  reads it back, so its ordering is a contract between two independent
  implementations. A transposed matrix, a bias attached to the wrong layer or a
  LayerNorm applied out of turn shows up here and nowhere else until a training
  run has quietly learned nothing. Several public rows, several configs per
  row, both seats, and the degenerate supports a real solve does produce: a
  query with no configs at all, a query with exactly one, and counts that are
  all zero.

* **Slot permutation.** Which slot the draft put a unit in is a pure
  relabelling — the ten coin types are described by their printed card facts,
  not by an identity embedding — so permuting the five slots of each player
  must leave every value and every auxiliary logit exactly where it was, as
  long as every place a slot index appears moves together. This is the check
  that caught raw belief marginals entering the join through a per-slot dense
  layer, which cost half the value signal. It runs against torch only: parity
  above carries it over to Rust.

    python train/test_parity.py
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import warchest  # noqa: E402
from value_net import Net  # noqa: E402

PUBFEAT = warchest.PUBFEAT
N_HEXES = warchest.N_HEXES
HEX_CH = warchest.HEX_CH
HEX_FACTS = warchest.HEX_FACTS
NTYPE = warchest.NTYPE
NSLOT = warchest.NSLOT
PILE_COUNTS = warchest.PILE_COUNTS
CARD_FEATS = warchest.CARD_FEATS
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
OFF_PILES = warchest.OFF_PILES
OFF_CARDS = warchest.OFF_CARDS
OFF_LOOSE = warchest.OFF_LOOSE

# (name, support size per canonical query, whether the counts are all zero).
CASES = [
    ("ragged supports", [3, 5, 1, 8, 2, 2, 6, 4, 7, 1, 5, 3], False),
    ("uniform supports", [4] * 8, False),
    ("singleton supports", [1] * 6, False),
    ("empty supports", [0, 4, 3, 0, 1, 1, 0, 0, 5, 2], False),
    ("one config in the batch", [0, 1], False),
    ("all-zero counts", [2, 3, 3, 2], True),
]


def random_net(seed):
    """A net whose every weight is visible in the output.

    The production init deliberately makes `cfg_f` tiny so that every value
    starts at the bias; leaving it there would let the readout half of the blob
    pass parity while being wired to the wrong place.
    """
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
    """Random rows in the frozen public encoding.

    Noise everywhere the network only ever sees as a float, but real occupancy:
    at most one coin type per hex, as an exact one-hot, because Rust reads the
    occupant by finding the first non-zero and torch reads it as a matmul.
    """
    x = rng.standard_normal((rows, PUBFEAT)).astype(np.float32)
    hexes = x[:, :N_HEXES * HEX_CH].reshape(rows, N_HEXES, HEX_CH)
    hexes[:, :, HEX_FACTS:] = 0
    occupant = rng.integers(0, NTYPE + 1, (rows, N_HEXES))
    r, h = np.nonzero(occupant < NTYPE)
    hexes[r, h, HEX_FACTS + occupant[r, h]] = 1
    return x


def belief(rng, sizes, zero_counts=False):
    """A ragged belief per canonical query: `seg`, `phi`, and weights that sum
    to one over each support, which is what a solve hands the network."""
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
    """The torch forward on numpy inputs: values and auxiliary logits."""
    with torch.no_grad():
        v, aux = net(torch.from_numpy(np.ascontiguousarray(xpub)),
                     torch.from_numpy(np.ascontiguousarray(phi)),
                     torch.from_numpy(weight),
                     torch.from_numpy(seg.astype(np.int64)), queries)
    return v.numpy(), aux.numpy()


def slot_permutation(perm):
    """The `PUBFEAT` permutation a relabelling of the draft slots induces.

    A slot index appears in the occupancy one-hot of every hex, in the pile
    block and in the card block; both players' blocks are permuted the same
    way, so the seat a type belongs to does not move.
    """
    full = np.concatenate([perm, NSLOT + perm])
    idx = np.arange(PUBFEAT)
    hexes = idx[:N_HEXES * HEX_CH].reshape(N_HEXES, HEX_CH)
    hexes[:, HEX_FACTS:] = hexes[:, HEX_FACTS:][:, full]
    piles = idx[OFF_PILES:OFF_CARDS].reshape(NTYPE, PILE_COUNTS)
    piles[:] = piles[full]
    cards = idx[OFF_CARDS:OFF_LOOSE].reshape(NTYPE, CARD_FEATS)
    cards[:] = cards[full]
    return idx


def blob_parity(net, rng):
    """Every case through both implementations of the same weights."""
    worst = 0.0
    for name, sizes, zero in CASES:
        queries = len(sizes)
        xpub = public_rows(rng, queries)
        seg, phi, weight = belief(rng, sizes, zero)
        want, _ = run(net, xpub, phi, weight, seg, queries)
        got = np.asarray(warchest.infer(
            xpub.ravel(), phi.ravel(), weight, seg, queries), np.float32)
        assert got.shape == want.shape, f"{name}: {got.shape} vs {want.shape}"
        scale = max(1.0, float(np.abs(want).max()))
        assert scale > 1e-2, f"{name}: values are all zero, the test proves nothing"
        err = float(np.max(np.abs(want - got)))
        assert err < 1e-4 * scale, f"{name}: value parity failure: {err:.3e}"
        worst = max(worst, err / scale)
        print(f"  {name:24s} {queries:3d} queries {len(seg):4d} configs "
              f"|v|<={scale:6.2f}  max err {err:.3e}")
    print(f"blob parity ok: worst relative error {worst:.3e}")


def slot_invariance(net, rng, perms=6):
    """Relabel the draft six ways; nothing the network says may move."""
    sizes = [4, 3, 6, 2, 5, 5]
    queries = len(sizes)
    xpub = public_rows(rng, queries)
    seg, phi, weight = belief(rng, sizes)
    base, base_aux = run(net, xpub, phi, weight, seg, queries)
    spread = float(base.std())
    assert spread > 1e-3, "degenerate invariance inputs"
    worst = 0.0
    for _ in range(perms):
        perm = rng.permutation(NSLOT)
        px = xpub[:, slot_permutation(perm)]
        pphi = phi.reshape(-1, 3, NSLOT)[:, :, perm].reshape(-1, CCOUNTS)
        got, got_aux = run(net, px, pphi, weight, seg, queries)
        dv = float(np.max(np.abs(got - base)))
        da = float(np.max(np.abs(got_aux - base_aux)))
        worst = max(worst, dv, da)
        print(f"  slots {perm}  values {dv:.2e}  aux {da:.2e}")
    # Relative to the size of the values, because these random weights put them
    # in the tens and an absolute 1e-5 would be two float32 ulps away. The
    # defect this guards against was half the value spread, so the margin is
    # three orders of magnitude either side.
    tol = 1e-4 * max(1.0, float(np.abs(base).max()))
    assert worst < tol, (f"slot permutation is not a relabelling: {worst:.3e} "
                         f"against a value spread of {spread:.3e}")
    print(f"slot invariance ok: worst {worst:.3e}, {worst / spread:.1e} of the "
          f"value spread ({spread:.3e})")


def main():
    rng = np.random.default_rng(11)
    net = random_net(7)
    net.push(0)
    blob_parity(net, rng)
    slot_invariance(net, rng)


if __name__ == "__main__":
    main()
