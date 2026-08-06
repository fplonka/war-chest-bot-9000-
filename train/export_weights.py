"""Dump a trained checkpoint into the flat binary the Rust tools read.

Layout (little-endian):
    u32 n_dims, then n_dims * u32 dims,
    u32 n_w,    then n_w    * f32 weights,
    u32 n_b,    then n_b    * f32 biases,
    u32 n_ln,   then n_ln   * f32 layernorm weight/bias per hidden layer.

Same ordering as `Mlp.push`, so a benchmark or an example measures exactly the
network the trainer ships to the workers.
"""

import struct
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from value_net import Mlp  # noqa: E402


def load(path):
    """A checkpoint as an `Mlp`, in the shape it was saved with.

    Checkpoints written before the policy head existed have no `wq`/`wk`/`wp`,
    so those are left at their random initialisation and `has_policy` is set to
    False. That is safe for anything that only searches — search asks the
    network for values and never for action probabilities, so an old snapshot
    plays exactly the moves it always did and its Elo stays comparable. It is
    not safe for anything that reads the policy, which is what the flag is for.
    """
    # Our own checkpoints; torch 2.6+ defaults to weights_only=True.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Mlp(ck["hidden"], ck.get("dg", 64), ck.get("rank", 64))
    missing, _ = net.load_state_dict(ck["value"], strict=False)
    net.has_policy = not any(k.startswith(("wq.", "wk.", "wp.")) for k in missing)
    return net


def main():
    src, dst = sys.argv[1], sys.argv[2]
    net = load(src)
    w, b, ln = net.flat()
    with open(dst, "wb") as f:
        f.write(struct.pack("<I", len(net.dims)))
        f.write(struct.pack(f"<{len(net.dims)}I", *net.dims))
        for a in (w, b, ln):
            a = np.ascontiguousarray(a, np.float32)
            f.write(struct.pack("<I", a.size))
            f.write(a.tobytes())
    print(f"wrote {dst}: dims={net.dims}")


if __name__ == "__main__":
    main()
