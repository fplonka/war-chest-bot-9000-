"""The network a pre-describer checkpoint was trained with.

Frozen, and eval-only. The card describer changed the public encoding's width
and layout, so a checkpoint from before it cannot read a row written after it —
and a gate is worthless if the new architecture cannot be played against the
pool it is meant to beat. This loads those checkpoints and ships their weights;
it never trains.

`dims` has five entries here against the current eight, which is how both Rust
(`Mlp::v1`) and `export_weights.load` tell the two apart.

Nothing here is maintained or extended. Delete this module, and `engine/src/v1.rs`
beside it, when the pool has rotated past every checkpoint that needs it.
"""

import numpy as np
import torch.nn as nn

import warchest

PUBFEAT_V1 = warchest.PUBFEAT_V1
CFEAT = warchest.CFEAT


class MlpV1(nn.Module):
    def __init__(self, hidden, dg=64, rank=64):
        super().__init__()
        self.dims = [PUBFEAT_V1, hidden, CFEAT, dg, rank]
        self.w0 = nn.Linear(PUBFEAT_V1, hidden)
        self.w1 = nn.Linear(hidden, hidden)
        self.wb = nn.Linear(2 * dg, hidden, bias=False)
        self.wc = nn.Linear(CFEAT, dg)
        self.wg = nn.Linear(dg, rank + 1)
        self.wu = nn.Linear(hidden, rank)
        self.ln0 = nn.LayerNorm(hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.has_policy = False

    def flat(self):
        """The order `Mlp::from_flat_v1` reads."""
        f = lambda a: np.ascontiguousarray(a, np.float32)
        w = f(np.concatenate([l.weight.detach().cpu().t().contiguous().numpy().ravel()
                              for l in (self.w0, self.w1, self.wb, self.wc, self.wg, self.wu)]))
        b = f(np.concatenate([l.bias.detach().cpu().numpy().ravel()
                              for l in (self.w0, self.w1, self.wc, self.wg, self.wu)]))
        ln = f(np.concatenate([t.detach().cpu().numpy().ravel()
                               for n in (self.ln0, self.ln1) for t in (n.weight, n.bias)]))
        return w, b, ln

    def push(self, slot):
        warchest.set_weights(self.dims, *self.flat(), slot)
