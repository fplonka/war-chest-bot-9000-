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
CCOUNTS = warchest.CCOUNTS
CARD_FEATS = warchest.CARD_FEATS
N_HEXES = warchest.N_HEXES
N_UNITS = warchest.N_UNITS
NSLOT = warchest.NSLOT
NTYPE = warchest.NTYPE
HEX_CH = warchest.HEX_CH
HEX_FACTS = warchest.HEX_FACTS
PILE_COUNTS = warchest.PILE_COUNTS
LOOSE = warchest.LOOSE
OFF_PILES = warchest.OFF_PILES
OFF_CARDS = warchest.OFF_CARDS
OFF_LOOSE = warchest.OFF_LOOSE
AOFF_PAYS = warchest.AOFF_PAYS
AUX = warchest.AUX


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

    def __init__(self, hidden, dg=64, rank=64, de=32, dc=64, head=None):
        super().__init__()
        # `head` is the width of the second public matrix, the belief
        # projection, the second LayerNorm and both readouts. It is
        # checkpoint metadata like every other width; `head == hidden` is
        # the network the head-width split was taken from.
        head = hidden if head is None else head
        self.dims = [PUBFEAT, hidden, head, CFEAT, dg, rank, AFEAT, de, dc]
        self.de = de
        self.head = head
        xdim = N_HEXES * (HEX_FACTS + de) + 2 * de + LOOSE
        # The card describer, and the pile summary that reads it. Everything
        # that names a card names a coin-type index into `e`. The describer
        # reads the card's rulebook facts (so related cards share learning)
        # and adds a learned per-unit identity embedding (so an individual
        # card can be memorised); both are needed because a draft the network
        # has never seen must still be describable.
        self.wd0 = nn.Linear(CARD_FEATS, dc)
        self.wd1 = nn.Linear(dc, de)
        self.wid = nn.Embedding(N_UNITS, de)
        self.wpile = nn.Linear(PILE_COUNTS + de, de)
        self.w0 = nn.Linear(xdim, hidden)
        self.w1 = nn.Linear(hidden, head)
        # The belief's connection into the hidden layer. No bias: it is added to
        # a layer that already has one.
        self.wb = nn.Linear(2 * dg, head, bias=False)
        # The holding tower is per coin type and summed over the five slots, so
        # it has no order and any draft fits. A residual MLP sits on the sum:
        # s = sum_k relu([counts_k, seat, e_k] Wc + bc) is the additive tower,
        # and z = s + relu(s Wh1 + bh1) Wh2 + bh2 lets the tower learn card
        # combinations. Wh2/bh2 start at zero, so the network begins exactly as
        # the additive one.
        self.wc = nn.Linear(4 + de, dg)
        self.wh1 = nn.Linear(dg, dg)
        self.wh2 = nn.Linear(dg, dg)
        self.wg = nn.Linear(dg, rank + 1)
        self.wu = nn.Linear(head, rank)
        # The policy head: an action tower, and the two halves of its readout.
        # Both towers are shared with the value, so this is three matrices.
        self.wq = nn.Linear(AFEAT + de, rank)
        self.wk = nn.Linear(dg, rank)
        self.wp = nn.Linear(head, rank)
        # Auxiliary heads, training only. Their targets are dense facts about how
        # the game actually went -- markers three rounds on, whether initiative
        # changes hands, the result -- so every row carries a different answer
        # and every row gives the shared layers a gradient the single value
        # number does not. Never in `flat()`, so the Rust play path never sees
        # them and they cost nothing at play time.
        self.aux = nn.Linear(head, AUX + 2)
        self.ln0 = nn.LayerNorm(hidden)
        self.ln1 = nn.LayerNorm(head)
        # Start near zero so the first bootstrapped targets are not dominated by
        # random leaf values.
        nn.init.zeros_(self.wg.bias)
        nn.init.normal_(self.wg.weight, std=1e-3)
        # The holding residual starts as the identity: zeroed second stage.
        nn.init.zeros_(self.wh2.weight)
        nn.init.zeros_(self.wh2.bias)
        # True only when the weights were actually trained. A checkpoint from
        # before the policy head exists loads with `strict=False` and leaves
        # these three matrices at their random initialisation, which plays fine
        # as long as nothing reads them -- search asks only for values. Anything
        # that does read them asserts on this rather than quietly playing noise
        # and reporting it as a strength result.
        self.has_policy = True

    def aux_loss(self, xpub, unit_ids, target):
        """The auxiliary heads' loss, on the same hidden layer the value reads.

        `target` is `[B, AUX]`: two marker counts three rounds on, the
        initiative-flip flag, and the result class. One matrix produces
        `AUX + 2` numbers, because the result is three classes rather than one:
        two regressions, one binary, one 3-way.

        Beliefs are not needed -- these are facts about the public future -- so
        the hidden layer is taken with an empty belief block, which keeps the
        gradient on the part of the trunk every row shares.
        """
        b = xpub.shape[0]
        e = self.cards(xpub, unit_ids)
        h = F.relu(self.ln0(self.w0(self.trunk_input(xpub, e))))
        zero = torch.zeros(b, self.wb.in_features, dtype=h.dtype, device=h.device)
        h = F.relu(self.ln1(self.w1(h) + self.wb(zero)))
        a = self.aux(h)
        return (F.mse_loss(a[:, :2], target[:, :2])
                + F.binary_cross_entropy_with_logits(a[:, 2], target[:, 2])
                + F.cross_entropy(a[:, 3:], target[:, 3].long()))

    def cards(self, xpub, unit_ids):
        """The card table `e`, `[B, NTYPE, de]` — one embedding per coin type.

        `unit_ids` is `[B, NTYPE]` (the row's stored ids, player-major slot
        order): the learned id embedding of each coin type is added to the
        facts' output.
        """
        c = xpub[:, OFF_CARDS:OFF_CARDS + NTYPE * CARD_FEATS]
        return (self.wd1(F.relu(self.wd0(c.reshape(-1, NTYPE, CARD_FEATS))))
                + self.wid(unit_ids))

    def trunk_input(self, xpub, e):
        """The trunk's input, assembled from a stored row and the card table.

        A row stores one-hots, not embeddings: the embedding is learned, so a
        stored row that held it would carry whichever weights were live when the
        row was written and would pass no gradient back to the describer. The
        gather is written as the one-hot matmul it is.
        """
        b = xpub.shape[0]
        hx = xpub[:, :N_HEXES * HEX_CH].reshape(b, N_HEXES, HEX_CH)
        hex_e = hx[:, :, HEX_FACTS:] @ e                             # [B, N_HEXES, de]
        piles = xpub[:, OFF_PILES:OFF_CARDS].reshape(b, NTYPE, PILE_COUNTS)
        p = F.relu(self.wpile(torch.cat([piles, e], -1)))             # [B, NTYPE, de]
        return torch.cat([
            hx[:, :, :HEX_FACTS].reshape(b, -1),
            hex_e.reshape(b, -1),
            p.reshape(b, 2, NSLOT, -1).sum(2).reshape(b, -1),
            xpub[:, OFF_LOOSE:OFF_LOOSE + LOOSE],
        ], -1)

    def holdings(self, phi, e):
        """The holding tower: per coin type, three counts and the seat alongside
        that card's embedding, through one shared matrix, summed over the five
        slots. `phi` is `[U, CFEAT]` and `e` the card table of each config's own
        row, `[U, NTYPE, de]`."""
        seat = phi[:, CCOUNTS].long()
        counts = phi[:, :CCOUNTS].reshape(-1, 3, NSLOT).transpose(1, 2)   # [U, NSLOT, 3]
        # The seat picks which half of the card table this holding's slots index.
        mine = e[torch.arange(e.shape[0], device=e.device).unsqueeze(1),
                 seat.unsqueeze(1) * NSLOT + torch.arange(NSLOT, device=e.device)]
        s = phi[:, CCOUNTS].reshape(-1, 1, 1).expand(-1, NSLOT, 1)
        # Rectify before the sum: a sum of raw linear maps is a linear map of
        # the sum, and the sum of the inputs has forgotten which count belongs
        # to which card -- the one thing this tower exists to remember.
        z = F.relu(self.wc(torch.cat([counts, s, mine], -1))).sum(1)
        return z + self.wh2(F.relu(self.wh1(z)))

    def actions(self, psi, e):
        """The action tower. `psi` is `[A, AFEAT]` and `e` the card table of each
        action's own row, `[A, NTYPE, de]`. The paying card's embedding is
        gathered through the coin-type one-hot `psi` already carries."""
        pay = psi[:, AOFF_PAYS:AOFF_PAYS + NTYPE].unsqueeze(1) @ e     # [A, 1, de]
        return F.relu(self.wq(torch.cat([psi, pay.squeeze(1)], -1)))

    def forward(self, xpub, unit_ids, phi, inv, w, seg, nseg):
        """Values for every config in a ragged batch.

        `xpub` is `[B, PUBFEAT]`. The configs of every row and player are
        concatenated into one list of length `N`; `w[i]` is config `i`'s belief
        probability and `seg[i] = 2 * row + player` says where it belongs.

        The holding tower runs over *distinct* configs only: `phi` is
        `[U, CFEAT]` and `inv` maps each of the `N` entries to its row in it. The
        Rust solver deduplicates for the same reason.
        """
        e = self.cards(xpub, unit_ids)
        # A distinct config belongs to a row, and a row to a game, so it reads
        # that game's card table. `seg // 2` is the row of each entry; the first
        # entry naming a distinct config fixes which table it uses.
        crow = torch.zeros(phi.shape[0], dtype=torch.long, device=phi.device)
        crow.scatter_(0, inv, seg // 2)
        z = self.holdings(phi, e[crow])
        g = self.wg(z)
        b = torch.zeros(nseg, z.shape[1], dtype=z.dtype, device=z.device)
        b.index_add_(0, seg, z[inv] * w.unsqueeze(1))
        h = F.relu(self.ln0(self.w0(self.trunk_input(xpub, e))))
        h = F.relu(self.ln1(self.w1(h) + self.wb(b.reshape(xpub.shape[0], -1))))
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
        w = f(np.concatenate(
            [l.weight.detach().cpu().t().contiguous().numpy().ravel()
             for l in (self.wd0, self.wd1)]
            + [self.wid.weight.detach().cpu().contiguous().numpy().ravel()]
            + [l.weight.detach().cpu().t().contiguous().numpy().ravel()
               for l in (self.wpile, self.w0, self.w1, self.wb, self.wc,
                         self.wh1, self.wh2, self.wg,
                         self.wu, self.wq, self.wk, self.wp)]))
        b = f(np.concatenate([l.bias.detach().cpu().numpy().ravel()
                              for l in (self.wd0, self.wd1, self.wpile,
                                        self.w0, self.w1, self.wc,
                                        self.wh1, self.wh2,
                                        self.wg, self.wu, self.wq, self.wk, self.wp)]))
        ln = f(np.concatenate([t.detach().cpu().numpy().ravel()
                               for n in (self.ln0, self.ln1) for t in (n.weight, n.bias)]))
        return w, b, ln

    def push(self, slot):
        """Ship weights to the Rust workers."""
        warchest.set_weights(self.dims, *self.flat(), slot)
