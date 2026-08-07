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

# The hex-neighbour encoder's fixed geometry: hex width, layer count, and the
# per-hex gather of (self, six neighbours) in the board's fixed direction
# order, off-board = the zero row (index N_HEXES).
HEX_W = 192
HEX_LAYERS = 6
_HEX_NB = np.asarray(warchest.hex_neighborhood(), dtype=np.int64).reshape(N_HEXES, 7)
# Normalized axial coordinates per hex, (x-3)/3, (y-3)/3.
_COORDS = np.zeros((N_HEXES, 2), np.float32)
for h, (x, y) in enumerate(warchest.hex_coords()):
    _COORDS[h] = ((x - 3) / 3.0, (y - 3) / 3.0)
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


class HexLayer(nn.Module):
    """One residual neighbour layer: every hex reads itself and its six
    neighbours in the board's fixed direction order (off-board = zero).

        y[h] = relu(layer_norm(x[h] + Wself x[h] + sum_d Wdir[d] x[nb(h,d)]))

    The self term is the plain residual plus Wself's contribution; the
    directions are fixed, so a stack of these can express the straight-line
    relations the unit cards are full of.
    """

    def __init__(self, w):
        super().__init__()
        self.wself = nn.Linear(w, w, bias=False)
        self.wdir = nn.ModuleList([nn.Linear(w, w, bias=False) for _ in range(6)])
        self.ln = nn.LayerNorm(w)

    def forward(self, x):
        # Off-board neighbours read the zero row (index N_HEXES).
        nb = F.pad(x, (0, 0, 0, 1))[:, _HEX_NB, :]  # [B, 37, 7, w]
        agg = self.wself(x)
        for d in range(6):
            agg = agg + self.wdir[d](nb[:, :, d + 1, :])
        return F.relu(self.ln(x + agg))


class Res(nn.Module):
    """One holding-tower residual block: `z + b(relu(a(z)))`. `b` starts at
    zero, so a fresh network is exactly the additive tower."""

    def __init__(self, dg):
        super().__init__()
        self.a = nn.Linear(dg, dg)
        self.b = nn.Linear(dg, dg)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)


# Renames from the fixed-layout checkpoints to the tower layout. Everything
# else kept its name.
_UPGRADE = {
    "wd0": "card.0", "wd1": "card.1",
    "w0": "pub.0", "ln0": "pub_ln.0", "w1": "pub_out",
    "wc": "slot_out", "wh1": "res.0.a", "wh2": "res.0.b",
}


def upgrade_state_dict(sd):
    """A fixed-layout state dict, renamed into the tower layout."""
    out = {}
    for k, v in sd.items():
        head, dot, rest = k.partition(".")
        out[_UPGRADE.get(head, head) + dot + rest] = v
    return out


class Mlp(nn.Module):
    """The value network: `v(PBS, config) -> scalar`.

    Two towers meeting in a bilinear readout. The config tower embeds one
    player's exact private state; the PBS tower embeds the public state and,
    through a belief-weighted sum of the *same* config embeddings, the belief.

        z(c) = holding tower over per-slot rows, summed     [dg]
        g(c) = z(c) Wg + bg                                 [rank + 1]
        e_p  = sum_c beta_p(c) z(c)                         [dg]
        h0   = public tower (h0 is pre-norm)                [head]
        h    = head: relu(LN1(h0 + [e_0|e_1] Wb)), extras   [head_out]
        u    = h Wu + bu                                    [rank]
        v(c) = <u, g(c)[:rank]> + g(c)[rank]

    The *wiring* is fixed — card table, set sums, belief sums, bilinear
    readout; it is the game-specific part. Every tower's depth and width is a
    constructor argument and checkpoint metadata:

      pub:  public LN+ReLU layer widths (default `[hidden]`)
      head: the h0 / LN1 width (default `hidden`)
      hmlp: extra plain-ReLU head layers after the entry (default none) —
            this is the per-iteration path, so depth here costs solve time
      card: card-describer hidden widths (default `[dc]`)
      slot: holding-tower hidden widths before the summed layer (default none)
      nres: holding residual blocks (default 1)

    `rank` is the one dimension chosen rather than inherited: every config
    costs a `rank`-long dot product per leaf per iteration. LayerNorm on the
    public layers and the head entry, as the reference does: the bootstrapped
    targets shift scale over training and the norms keep activations stable.
    """

    def __init__(self, hidden, dg=64, rank=64, de=32, dc=64, head=None, hex_net=False,
                 no_id=False, no_facts=False, no_residual=False,
                 pub=None, hmlp=None, card=None, slot=None, nres=1):
        # The 7a screening variants (`no_*`) switch parts off; screening-only.
        self.no_id = no_id
        self.no_facts = no_facts
        if no_residual:
            nres = 0
        super().__init__()
        pub = list(pub) if pub else [hidden]
        hmlp = list(hmlp) if hmlp else []
        card = list(card) if card else [dc]
        slot = list(slot) if slot else []
        head = head or hidden
        self.hex_net = hex_net
        self.de, self.dg, self.rank, self.head_in, self.nres = de, dg, rank, head, nres
        self.pub_w, self.hmlp_w, self.card_w, self.slot_w = pub, hmlp, card, slot
        head_out = hmlp[-1] if hmlp else head

        def chain(first, mid, last):
            widths = [first, *mid, last]
            return nn.ModuleList(
                [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])

        # The card describer: rulebook facts through a ReLU chain, plus a
        # learned per-unit identity embedding. Both paths, so related cards
        # share learning and an individual card can still be memorised.
        self.card = chain(CARD_FEATS, card, de)
        self.wid = nn.Embedding(N_UNITS, de)
        self.wpile = nn.Linear(PILE_COUNTS + de, de)
        xdim = N_HEXES * (HEX_FACTS + de) + 2 * de + LOOSE
        if hex_net:
            # The hex trunk replaces the first public layer; offline-only.
            assert len(pub) == 1, "hex encoder supports a single public layer"
            self.hex_in = nn.Linear(HEX_FACTS + de + 2, HEX_W)
            self.hex_layers = nn.ModuleList([HexLayer(HEX_W) for _ in range(HEX_LAYERS)])
            self.w_combine = nn.Linear(HEX_W + 2 * de + LOOSE, pub[0])
            self.pub = nn.ModuleList([])
        else:
            widths = [xdim, *pub]
            self.pub = nn.ModuleList(
                [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.pub_ln = nn.ModuleList([nn.LayerNorm(w) for w in pub])
        self.pub_out = nn.Linear(pub[-1], head)
        # The belief's connection into the head. No bias: it is added to a
        # layer that already has one (pub_out's, folded into ln1's pass).
        self.wb = nn.Linear(2 * dg, head, bias=False)
        self.ln1 = nn.LayerNorm(head)
        widths = [head, *hmlp]
        self.hmlp = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.wu = nn.Linear(head_out, rank)
        # The holding tower: per coin type through a shared chain, rectified,
        # summed over the five slots (a sum has no order, so any draft fits),
        # then the residual blocks.
        widths = [4 + de, *slot]
        self.slot = nn.ModuleList(
            [nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.slot_out = nn.Linear(slot[-1] if slot else 4 + de, dg)
        self.res = nn.ModuleList([Res(dg) for _ in range(nres)])
        self.wg = nn.Linear(dg, rank + 1)
        # The policy head: an action tower and the two halves of its readout.
        self.wq = nn.Linear(AFEAT + de, rank)
        self.wk = nn.Linear(dg, rank)
        self.wp = nn.Linear(head_out, rank)
        # Auxiliary heads, training only. Never in `flat()`.
        self.aux = nn.Linear(head_out, AUX + 2)
        # Start the readout near zero so the first bootstrapped targets are
        # not dominated by random leaf values.
        nn.init.zeros_(self.wg.bias)
        nn.init.normal_(self.wg.weight, std=1e-3)
        self.has_policy = True

    def spec(self):
        """Constructor arguments, for the checkpoint."""
        return {"hidden": self.pub_w[-1], "dg": self.dg, "rank": self.rank,
                "de": self.de, "head": self.head_in, "hex_net": self.hex_net,
                "pub": self.pub_w, "hmlp": self.hmlp_w, "card": self.card_w,
                "slot": self.slot_w, "nres": self.nres}

    @property
    def dims(self):
        """The v3 shape vector `engine/src/net.rs::from_flat_v3` reads:
        `[3, de, dg, rank, head, nres]` then the four length-prefixed width
        lists (card, pub, hmlp, slot). A hex encoder cannot run in Rust, so
        it poisons the tag."""
        d = [4 if self.hex_net else 3, self.de, self.dg, self.rank, self.head_in, self.nres]
        for lst in (self.card_w, self.pub_w, self.hmlp_w, self.slot_w):
            d.append(len(lst))
            d.extend(lst)
        return d

    def aux_loss(self, xpub, unit_ids, target):
        """The auxiliary heads' loss, on the same hidden layer the value
        reads, taken with an empty belief block so the gradient lands on the
        shared trunk. `target` is `[B, AUX]`: two marker counts three rounds
        on, the initiative flip, the result class."""
        b = xpub.shape[0]
        e = self.cards(xpub, unit_ids)
        h0 = self._public(xpub, e)
        zero = torch.zeros(b, self.wb.in_features, dtype=h0.dtype, device=h0.device)
        a = self.aux(self._head(h0, zero))
        return (F.mse_loss(a[:, :2], target[:, :2])
                + F.binary_cross_entropy_with_logits(a[:, 2], target[:, 2])
                + F.cross_entropy(a[:, 3:], target[:, 3].long()))

    def cards(self, xpub, unit_ids):
        """The card table `e`, `[B, NTYPE, de]` — one embedding per coin type."""
        c = xpub[:, OFF_CARDS:OFF_CARDS + NTYPE * CARD_FEATS]
        e = torch.zeros(xpub.shape[0], NTYPE, self.de, dtype=xpub.dtype, device=xpub.device)
        if not self.no_facts:
            t = c.reshape(-1, NTYPE, CARD_FEATS)
            for lin in self.card[:-1]:
                t = F.relu(lin(t))
            e = self.card[-1](t)
        if not self.no_id:
            e = e + self.wid(unit_ids)
        return e

    def hex_input(self, xpub, e):
        """The hex trunk's input; offline-only. See `HexLayer`."""
        b = xpub.shape[0]
        hx = xpub[:, :N_HEXES * HEX_CH].reshape(b, N_HEXES, HEX_CH)
        facts = hx[:, :, :HEX_FACTS]
        emb = hx[:, :, HEX_FACTS:] @ e                       # [B, 37, de]
        coords = torch.as_tensor(_COORDS, dtype=xpub.dtype, device=xpub.device)
        x = F.relu(self.hex_in(torch.cat([facts, emb, coords.expand(b, -1, -1)], -1)))
        for layer in self.hex_layers:
            x = layer(x)
        piles = xpub[:, OFF_PILES:OFF_CARDS].reshape(b, NTYPE, PILE_COUNTS)
        p = F.relu(self.wpile(torch.cat([piles, e], -1))).reshape(b, 2, NSLOT, -1).sum(2).reshape(b, -1)
        return torch.cat([x.sum(1), p, xpub[:, OFF_LOOSE:OFF_LOOSE + LOOSE]], -1)

    def public_trunk(self, xpub, e):
        """The first public layer's pre-activation: `pub.0` over the flat
        input, or the hex encoder's combine."""
        if self.hex_net:
            return self.w_combine(self.hex_input(xpub, e))
        return self.pub[0](self.trunk_input(xpub, e))

    def trunk_input(self, xpub, e):
        """The trunk's input, assembled from a stored row and the card table.
        A row stores one-hots, not embeddings: the embedding is learned, so a
        stored row that held it would go stale. The gather is written as the
        one-hot matmul it is."""
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

    def _public(self, xpub, e):
        """The public tower up to `h0` (pre-norm): every layer
        `relu(LN(lin(x)))`, then the projection. `pub_out`'s bias rides into
        `ln1` at the head entry, matching the Rust split."""
        t = F.relu(self.pub_ln[0](self.public_trunk(xpub, e)))
        for lin, ln in zip(self.pub[1:], self.pub_ln[1:]):
            t = F.relu(ln(lin(t)))
        return self.pub_out(t)

    def _head(self, h0, b):
        """The per-iteration head: entry norm plus the extra ReLU layers."""
        t = F.relu(self.ln1(h0 + self.wb(b)))
        for lin in self.hmlp:
            t = F.relu(lin(t))
        return t

    def holdings(self, phi, e):
        """The holding tower: per coin type, three counts and the seat
        alongside that card's embedding, through the shared chain, rectified,
        summed over the five slots, then the residual blocks. `phi` is
        `[U, CFEAT]`, `e` the card table of each config's row `[U, NTYPE, de]`."""
        seat = phi[:, CCOUNTS].long()
        counts = phi[:, :CCOUNTS].reshape(-1, 3, NSLOT).transpose(1, 2)   # [U, NSLOT, 3]
        mine = e[torch.arange(e.shape[0], device=e.device).unsqueeze(1),
                 seat.unsqueeze(1) * NSLOT + torch.arange(NSLOT, device=e.device)]
        s = phi[:, CCOUNTS].reshape(-1, 1, 1).expand(-1, NSLOT, 1)
        x = torch.cat([counts, s, mine], -1)
        for lin in self.slot:
            x = F.relu(lin(x))
        # Rectify before the sum: a sum of raw linear maps forgets which
        # count belongs to which card.
        z = F.relu(self.slot_out(x)).sum(1)
        for blk in self.res:
            z = z + blk.b(F.relu(blk.a(z)))
        return z

    def actions(self, psi, e):
        """The action tower. `psi` is `[A, AFEAT]`, `e` the card table of each
        action's row; the paying card's embedding is gathered through the
        one-hot `psi` already carries."""
        pay = psi[:, AOFF_PAYS:AOFF_PAYS + NTYPE].unsqueeze(1) @ e     # [A, 1, de]
        return F.relu(self.wq(torch.cat([psi, pay.squeeze(1)], -1)))

    def forward(self, xpub, unit_ids, phi, inv, w, seg, nseg):
        """Values for every config in a ragged batch. See the docstring of the
        original for the layout; unchanged."""
        e = self.cards(xpub, unit_ids)
        crow = torch.zeros(phi.shape[0], dtype=torch.long, device=phi.device)
        crow.scatter_(0, inv, seg // 2)
        z = self.holdings(phi, e[crow])
        g = self.wg(z)
        b = torch.zeros(nseg, z.shape[1], dtype=z.dtype, device=z.device)
        b.index_add_(0, seg, z[inv] * w.unsqueeze(1))
        h0 = self._public(xpub, e)
        h = self._head(h0, b.reshape(xpub.shape[0], -1))
        u = self.wu(h)
        rk = u.shape[1]
        gc = g[inv]
        return (u[seg // 2] * gc[:, :rk]).sum(-1) + gc[:, rk]

    def flat(self):
        """The weights as the three flat arrays Rust reads, in the v3 order
        of `engine/src/net.rs::from_flat_v3`: every matrix row-major
        `[in, out]`, then every bias, then the LayerNorms. Nothing else knows
        this order; `train/test_parity.py` checks it."""
        f = lambda a: np.ascontiguousarray(a, np.float32)
        t = lambda l: l.weight.detach().cpu().t().contiguous().numpy().ravel()
        bias = lambda l: l.bias.detach().cpu().numpy().ravel()
        lins = list(self.card) + [None, self.wpile] + list(self.pub) + [self.pub_out] \
            + [None] + list(self.hmlp) + [self.wu] + list(self.slot) + [self.slot_out]
        for blk in self.res:
            lins += [blk.a, blk.b]
        lins += [self.wg, self.wq, self.wk, self.wp]
        w, b = [], []
        for l in lins:
            if l is None:
                continue
            w.append(t(l))
            b.append(bias(l))
        # wid after the card chain, wb after pub_out — weights only, no bias.
        w.insert(len(self.card), self.wid.weight.detach().cpu().contiguous().numpy().ravel())
        w.insert(len(self.card) + 2 + len(self.pub) + 1,
                 self.wb.weight.detach().cpu().t().contiguous().numpy().ravel())
        ln = []
        for n in list(self.pub_ln) + [self.ln1]:
            ln += [n.weight.detach().cpu().numpy().ravel(), n.bias.detach().cpu().numpy().ravel()]
        return f(np.concatenate(w)), f(np.concatenate(b)), f(np.concatenate(ln))

    def push(self, slot):
        """Ship weights to the Rust workers."""
        warchest.set_weights(self.dims, *self.flat(), slot)
