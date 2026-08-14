"""Parity between the PyTorch and Rust v4 value forwards."""

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import warchest  # noqa: E402
from value_net import Mlp  # noqa: E402


def public_rows(rng, rows):
    x = rng.standard_normal((rows, warchest.PUBFEAT)).astype(np.float32)
    hexes = x[:, :warchest.N_HEXES * warchest.HEX_CH].reshape(
        rows, warchest.N_HEXES, warchest.HEX_CH)
    hexes[:, :, warchest.HEX_FACTS:] = 0
    occupants = rng.integers(0, warchest.NTYPE + 1, (rows, warchest.N_HEXES))
    r, h = np.nonzero(occupants < warchest.NTYPE)
    hexes[r, h, warchest.HEX_FACTS + occupants[r, h]] = 1
    ids = rng.integers(0, warchest.N_UNITS, (rows, warchest.NTYPE), dtype=np.uint8)
    return x, ids


def main():
    torch.manual_seed(7)
    rng = np.random.default_rng(11)
    net = Mlp()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.normal_(0, 0.08)
        for norm in [*net.public_norm, net.belief_norm, net.candidate_norm,
                     *net.context_norm]:
            norm.weight.normal_(1, 0.2)
            norm.bias.normal_(0, 0.1)
    net.push(0)

    queries = 12
    xpub, ids = public_rows(rng, queries)
    sizes = rng.integers(1, 6, queries)
    seg = np.repeat(np.arange(queries, dtype=np.uint32), sizes)
    phi = rng.integers(0, 5, (len(seg), warchest.CCOUNTS)).astype(np.float32) / warchest.CNORM
    weight = np.concatenate([
        (lambda x: x / x.sum())(rng.random(n).astype(np.float32)) for n in sizes
    ])

    args = (
        torch.from_numpy(xpub),
        torch.from_numpy(ids.astype(np.int64)),
        torch.from_numpy(phi),
        torch.from_numpy(weight),
        torch.from_numpy(seg.astype(np.int64)),
        queries,
    )
    with torch.no_grad():
        expected = net(*args).numpy()
    got = np.asarray(warchest.infer(
        xpub.ravel(), ids.ravel(), phi.ravel(), weight, seg, queries), np.float32)
    error = float(np.max(np.abs(expected - got)))
    assert expected.std() > 1e-3, "degenerate parity inputs"
    assert error < 3e-4, f"value parity failure: {error:.3e}"
    print(f"value parity ok: max error {error:.3e}")


if __name__ == "__main__":
    main()
