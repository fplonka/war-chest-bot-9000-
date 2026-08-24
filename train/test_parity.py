"""Parity between the PyTorch and Rust forwards, slot invariance, and complete
public-input coverage.

* **Blob parity.** `Net.flat()` writes the flat weight blob and `NetLayout::new`
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
  must leave every value exactly where it was, as long as every place a slot
  index appears moves together. This is the check
  that caught raw belief marginals entering the join through a per-slot dense
  layer, which cost half the value signal. It runs against torch only: parity
  above carries it over to Rust.

* **Off-board piles.** A drafted unit's public piles matter before its first
  deployment. Changing those counts must move the value even when no matching
  coin occupies a hex.

    python train/test_parity.py
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import warchest  # noqa: E402
from value_net import AFEAT, Net  # noqa: E402

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
    """The torch forward on numpy inputs."""
    with torch.no_grad():
        v = net(torch.from_numpy(np.ascontiguousarray(xpub)),
                torch.from_numpy(np.ascontiguousarray(phi)),
                torch.from_numpy(weight),
                torch.from_numpy(seg.astype(np.int64)), queries)
    return v.numpy()


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
        want = run(net, xpub, phi, weight, seg, queries)
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


def policy_parity(net, rng):
    """The policy readout through both implementations of the same weights.

    `logit(c, a) = <cfg_p(c), e(a, h)>`, so this exercises the third config
    head and every action projection, including a nonzero belief join row.
    """
    sizes = [4, 3, 6, 2]
    queries = len(sizes)
    na = 7
    xpub = public_rows(rng, queries)
    seg, phi, weight = belief(rng, sizes)
    n = len(seg)

    # One-hot blocks exactly as `Net::action_feats` writes them: kind, the coin
    # slot spent (the last column meaning none), then the three squares.
    feat = np.zeros((na, AFEAT), np.float32)
    for a in range(na):
        feat[a, rng.integers(N_KINDS)] = 1.0
        feat[a, N_KINDS + rng.integers(NSLOT + 1)] = 1.0
        at = N_KINDS + NSLOT + 1
        for _ in range(3):
            feat[a, at + rng.integers(N_HEXES + 1)] = 1.0
            at += N_HEXES + 1

    cfg = rng.integers(0, n, size=24).astype(np.uint32)
    act = rng.integers(0, na, size=24).astype(np.uint32)

    with torch.no_grad():
        cards = net.cards(torch.from_numpy(np.ascontiguousarray(xpub)))
        physical = torch.from_numpy(np.ascontiguousarray(xpub))[0::2]
        board = net.board(physical, net.tokens(physical, cards[0::2]))
        tseg = torch.from_numpy(seg.astype(np.int64))
        _f, g, fp = net.configs(torch.from_numpy(np.ascontiguousarray(phi)),
                                cards[:, :NSLOT], tseg)
        h = net.heads(board, g, torch.from_numpy(weight), tseg, queries)
        assert float(h.abs().max()) > 1e-2, "policy belief head is zero"
        want = np.zeros(len(cfg), np.float32)
        for k in range(len(cfg)):
            query = int(seg[cfg[k]])
            row = query // 2
            e = net.actions(torch.from_numpy(feat[act[k]:act[k] + 1]),
                            board[row:row + 1], h[query:query + 1])
            want[k] = float((fp[cfg[k]] * e[0]).sum())

    got = np.asarray(warchest.infer_policy(
        xpub.ravel(), phi.ravel(), weight, seg, feat.ravel(), cfg, act, queries),
        np.float32)
    assert got.shape == want.shape, f"policy: {got.shape} vs {want.shape}"
    scale = max(1.0, float(np.abs(want).max()))
    assert float(np.abs(want).max()) > 1e-2, "policy logits are all zero"
    err = float(np.max(np.abs(want - got)))
    assert err < 1e-4 * scale, f"policy parity failure: {err:.3e}"
    print(f"policy parity ok: {len(cfg)} cells over {na} actions, "
          f"|logit|<={scale:.2f}, max err {err:.3e}")


def slot_invariance(net, rng, perms=6):
    """Relabel the draft six ways; nothing the network says may move."""
    sizes = [4, 3, 6, 2, 5, 5]
    queries = len(sizes)
    xpub = public_rows(rng, queries)
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
    # Relative to the size of the values, because these random weights put them
    # in the tens and an absolute 1e-5 would be two float32 ulps away. The
    # defect this guards against was half the value spread, so the margin is
    # three orders of magnitude either side.
    tol = 1e-4 * max(1.0, float(np.abs(base).max()))
    assert worst < tol, (f"slot permutation is not a relabelling: {worst:.3e} "
                         f"against a value spread of {spread:.3e}")
    print(f"slot invariance ok: worst {worst:.3e}, {worst / spread:.1e} of the "
          f"value spread ({spread:.3e})")

def offboard_pile_visibility(net, rng):
    """Every type's public piles reach the trunk, occupied or not."""
    sizes = [1, 1]
    xpub = public_rows(rng, len(sizes))
    hexes = xpub[:, :N_HEXES * HEX_CH].reshape(len(sizes), N_HEXES, HEX_CH)
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
    """The Python entry point matches the Rust encoder on real mirrored rows."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requires a working GPU")
    rows = np.frombuffer(bytes(warchest.mirror_row_pairs(128, 19)), np.uint8)
    rows = rows.reshape(-1, warchest.ROW_BYTES)[:4096].copy()
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
    blob_parity(net, rng)
    policy_parity(net, rng)
    slot_invariance(net, rng)
    offboard_pile_visibility(net, rng)
    packed_row_cuda_parity()


if __name__ == "__main__":
    main()
