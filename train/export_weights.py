"""Dump a trained checkpoint into the flat binary the Rust bench harness reads.

Layout (little-endian):
    u32 n_dims, then n_dims * u32 dims,
    u32 n_w,    then n_w    * f32 weights (per layer, row-major [in, out]),
    u32 n_b,    then n_b    * f32 biases,
    u32 n_ln,   then n_ln   * f32 layernorm weight/bias per hidden layer.

Same ordering as `Mlp.push`, so the bench measures exactly the network the
trainer ships to the workers.
"""

import struct
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from train import Mlp  # noqa: E402

import warchest  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu")
    net = Mlp(warchest.FEAT, ck["hidden"], 2 * warchest.NHAND)
    sd = ck["value"]
    if sd["lin.0.weight"].shape[1] == warchest.FEAT:
        # A checkpoint from before the belief block moved to the second hidden
        # layer. Rewiring it is exact in shape (the parameter count is the same
        # either way) and gives the benchmark a network that plays plausibly.
        w0 = sd.pop("lin.0.weight")
        sd["lin.0.weight"] = w0[:, : net.dims[0]].contiguous()
        sd["bel.weight"] = w0[:, net.dims[0]:].contiguous()
    net.load_state_dict(sd)
    w = np.concatenate([l.weight.detach().t().contiguous().numpy().ravel()
                        for l in list(net.lin) + [net.bel]])
    b = np.concatenate([l.bias.detach().numpy().ravel() for l in net.lin])
    ln = np.concatenate([t.detach().numpy().ravel() for n in net.norm for t in (n.weight, n.bias)])
    with open(dst, "wb") as f:
        f.write(struct.pack("<I", len(net.dims)))
        f.write(struct.pack(f"<{len(net.dims)}I", *net.dims))
        f.write(struct.pack("<I", net.split))
        for a in (w, b, ln):
            a = np.ascontiguousarray(a, np.float32)
            f.write(struct.pack("<I", a.size))
            f.write(a.tobytes())
    print(f"wrote {dst}: dims={net.dims} split={net.split}")


if __name__ == "__main__":
    main()
