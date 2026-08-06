"""The value network, `v(PBS, config) -> scalar`.

Its own module because five things need it and only one of them trains: the
trainer, the offline architecture fitter, the weight exporter, the Rust/PyTorch
parity test, and the Elo ladder, which loads a checkpoint per snapshot.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
AFEAT = warchest.AFEAT


class Mlp(nn.Module):
    """The value network: `v(PBS, config) -> scalar`.

    Two towers. The config tower embeds one player's exact private state; the
    PBS tower embeds the public state and, through a belief-weighted sum of the
    *same* config embeddings, the belief. The value is their inner product.

        z(c) = relu(phi(c) Wc + bc)                 config embedding   [dg]
        g(c) = z(c) Wg + bg                         readout embedding  [r + 1]
        e_p  = sum_c beta_p(c) z(c)                 belief             [dg]
        h    = relu(LN(relu(LN(x W0 + b0)) W1 + b1 + [e_0; e_1] Wb))
        u    = h Wu + bu                            PBS readout        [r]
        v(c) = <u, g(c)[:r]> + g(c)[r]

    This is `csrc/liars_dice`'s shape with its two fixed-width private-state
    dimensions replaced by learned functions of the private state, because War
    Chest's private states do not fit in a fixed-width table. Set `g` to a
    one-hot lookup and the two are the same network.

    `rank` is the one dimension that has to be chosen rather than inherited: the
    reference gets `rank = hidden` for free because its readout is a lookup,
    while here every config costs a `rank`-long dot product. A config is
    sixteen numbers, so 64 is not a binding constraint on what the value can
    depend on, and it is 6x less per-config work than the hidden width.

    LayerNorm on every hidden layer, as the reference does (`use_layer_norm:
    true`): the raw features include unbounded-ish coin counts and the
    bootstrapped targets shift scale over training, so normalising between the
    affine and the activation is what keeps the hidden distribution stable as
    the target distribution moves.
    """

    def __init__(self, hidden, dg=64, rank=64):
        super().__init__()
        self.dims = [PUBFEAT, hidden, CFEAT, dg, rank, AFEAT]
        self.w0 = nn.Linear(PUBFEAT, hidden)
        self.w1 = nn.Linear(hidden, hidden)
        # The belief's connection into the hidden layer. No bias: it is added to
        # a layer that already has one.
        self.wb = nn.Linear(2 * dg, hidden, bias=False)
        self.wc = nn.Linear(CFEAT, dg)
        self.wg = nn.Linear(dg, rank + 1)
        self.wu = nn.Linear(hidden, rank)
        # The policy head: an action tower, and the two halves of its readout.
        # Both towers are shared with the value, so this is three matrices.
        self.wq = nn.Linear(AFEAT, rank)
        self.wk = nn.Linear(dg, rank)
        self.wp = nn.Linear(hidden, rank)
        self.ln0 = nn.LayerNorm(hidden)
        self.ln1 = nn.LayerNorm(hidden)
        # Start near zero so the first bootstrapped targets are not dominated by
        # random leaf values.
        nn.init.zeros_(self.wg.bias)
        nn.init.normal_(self.wg.weight, std=1e-3)
        # True only when the weights were actually trained. A checkpoint from
        # before the policy head exists loads with `strict=False` and leaves
        # these three matrices at their random initialisation, which plays fine
        # as long as nothing reads them -- search asks only for values. Anything
        # that does read them asserts on this rather than quietly playing noise
        # and reporting it as a strength result.
        self.has_policy = True

    def forward(self, xpub, phi, inv, w, seg, nseg):
        """Values for every config in a ragged batch.

        `xpub` is `[B, PUBFEAT]`. The configs of every row and player are
        concatenated into one list of length `N`; `w[i]` is config `i`'s belief
        probability and `seg[i] = 2 * row + player` says where it belongs.

        The config tower runs over *distinct* configs only: `phi` is `[U, CFEAT]`
        and `inv` maps each of the `N` entries to its row in it. A batch of 1024
        positions carries ~50k configs drawn from a couple of thousand distinct
        ones, and the readout embedding is `dg x (hidden + 1)` — by far the
        widest matmul here if it runs per entry. The Rust solver deduplicates
        for the same reason.
        """
        z = F.relu(self.wc(phi))
        g = self.wg(z)
        # The belief: a weighted sum of config embeddings, per (row, player).
        e = torch.zeros(nseg, z.shape[1], dtype=z.dtype, device=z.device)
        e.index_add_(0, seg, z[inv] * w.unsqueeze(1))
        h = F.relu(self.ln0(self.w0(xpub)))
        h = F.relu(self.ln1(self.w1(h) + self.wb(e.reshape(xpub.shape[0], -1))))
        u = self.wu(h)
        rk = u.shape[1]
        gc = g[inv]
        return (u[seg // 2] * gc[:, :rk]).sum(-1) + gc[:, rk]

    def flat(self):
        """The weights as the three flat arrays Rust reads: every matrix
        row-major `[in, out]`, then every bias, then the LayerNorms.

        The order here is `Mlp::from_flat`'s and nothing else knows it. Both
        ways of getting weights into Rust — `push` for a live run and
        `export_weights.py` for the offline tools — go through this, so there is
        one place for the two sides to agree and `test_parity.py` checks it.
        """
        f = lambda a: np.ascontiguousarray(a, np.float32)
        w = f(np.concatenate([l.weight.detach().cpu().t().contiguous().numpy().ravel()
                              for l in (self.w0, self.w1, self.wb, self.wc, self.wg,
                                        self.wu, self.wq, self.wk, self.wp)]))
        b = f(np.concatenate([l.bias.detach().cpu().numpy().ravel()
                              for l in (self.w0, self.w1, self.wc, self.wg, self.wu,
                                        self.wq, self.wk, self.wp)]))
        ln = f(np.concatenate([t.detach().cpu().numpy().ravel()
                               for n in (self.ln0, self.ln1) for t in (n.weight, n.bias)]))
        return w, b, ln

    def push(self, slot):
        """Ship weights to the Rust workers."""
        warchest.set_weights(self.dims, *self.flat(), slot)
