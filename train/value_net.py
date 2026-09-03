
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest

N_HEXES = warchest.N_HEXES
NTYPE = warchest.NTYPE
NSLOT = warchest.NSLOT
HEX_CH = warchest.HEX_CH
HEX_FACTS = warchest.HEX_FACTS
PILE_COUNTS = warchest.PILE_COUNTS
CARD_FEATS = warchest.CARD_FEATS
CCOUNTS = warchest.CCOUNTS
OFF_PILES = warchest.OFF_PILES
OFF_CARDS = warchest.OFF_CARDS
OFF_LOOSE = warchest.OFF_LOOSE
LOOSE = warchest.LOOSE

TYPE = 64
C = 96
BLOCKS = 8
D = 256
POOL = 64
CFGH = 128
JW = 128
JBLOCKS = 3
N_KINDS = warchest.N_KINDS
ACT_BYTES = warchest.ACT_BYTES

JOIN_IN = 2 * POOL + 1


def gelu(x):
    return F.gelu(x, approximate="tanh")


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.card1 = nn.Linear(CARD_FEATS, TYPE)
        self.card2 = nn.Linear(TYPE, TYPE)
        self.pile = nn.Linear(PILE_COUNTS, TYPE, bias=False)
        self.seat = nn.Embedding(2, TYPE)

        self.hex_stem = nn.Linear(HEX_FACTS, C)
        self.tok_stem = nn.Linear(TYPE, C, bias=False)
        self.pos = nn.Embedding(N_HEXES, C)
        self.glob_stem = nn.Linear(LOOSE, C, bias=False)

        self.blk1 = nn.ModuleList([nn.Linear(2 * C, C) for _ in range(BLOCKS)])
        self.blkg = nn.ModuleList([nn.Linear(2 * C, C) for _ in range(BLOCKS)])
        self.blk2 = nn.ModuleList([nn.Linear(C, C) for _ in range(BLOCKS)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(C) for _ in range(BLOCKS)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(C) for _ in range(BLOCKS)])
        self.ln_trunk = nn.LayerNorm(C)

        self.board_out = nn.Linear(2 * C + LOOSE, D)

        self.cfg1 = nn.Linear(3 + TYPE, CFGH)
        self.ln_cfg = nn.LayerNorm(CFGH)
        self.cfg_f = nn.Linear(CFGH, D)
        self.cfg_g = nn.Linear(CFGH, POOL)
        self.cfg_m = nn.Linear(TYPE, 3 * POOL, bias=False)
        self.cfg_p = nn.Linear(CFGH, D)
        self.act_kind = nn.Embedding(N_KINDS, C)
        self.act_role = nn.Embedding(5, C)
        nn.init.normal_(self.act_kind.weight, std=C ** -0.5)
        nn.init.ones_(self.act_role.weight)
        self.act_board = nn.Linear(D, C, bias=False)
        self.act_h = nn.Linear(D, C, bias=False)
        self.ln_act = nn.LayerNorm(C)
        self.act_out = nn.Linear(C, D)

        self.join_p = nn.Linear(D, JW, bias=False)
        self.join_b = nn.Linear(JOIN_IN, JW)
        self.joinw = nn.ModuleList([nn.Linear(JW, JW) for _ in range(JBLOCKS)])
        self.ln_join = nn.ModuleList([nn.LayerNorm(JW) for _ in range(JBLOCKS)])
        self.ln_jout = nn.LayerNorm(JW)
        self.join_out = nn.Linear(JW, D)
        self.ln_h = nn.LayerNorm(D)
        self.value_bias = nn.Parameter(torch.zeros(1))
        self.control = nn.Linear(C, 3)

        nn.init.normal_(self.cfg_f.weight, std=1e-3)
        nn.init.zeros_(self.cfg_f.bias)

        nb = torch.as_tensor(warchest.hex_neighbours(), dtype=torch.long)
        self.register_buffer("nb", nb.view(N_HEXES, 6), persistent=False)
        self.register_buffer("seat_of", torch.arange(NTYPE) // NSLOT,
                             persistent=False)


    def cards(self, xpub):
        facts = xpub[:, OFF_CARDS:OFF_CARDS + NTYPE * CARD_FEATS]
        facts = facts.reshape(-1, NTYPE, CARD_FEATS)
        return self.card2(gelu(self.card1(facts)))

    def tokens(self, xpub, cards):
        piles = xpub[:, OFF_PILES:OFF_CARDS].reshape(-1, NTYPE, PILE_COUNTS)
        return cards + self.pile(piles) + self.seat(self.seat_of)

    def trunk(self, xpub, projected):
        batch = xpub.shape[0]
        hexes = xpub[:, :N_HEXES * HEX_CH].reshape(batch, N_HEXES, HEX_CH)
        occupant = hexes[:, :, HEX_FACTS:] @ projected
        type_pool = gelu(projected).mean(1)
        loose = xpub[:, OFF_LOOSE:OFF_LOOSE + LOOSE]
        x = (self.hex_stem(hexes[:, :, :HEX_FACTS])
             + occupant
             + self.pos.weight
             + self.glob_stem(loose).unsqueeze(1)
             + type_pool.unsqueeze(1))
        for i in range(BLOCKS):
            a = gelu(self.ln1[i](x))
            pad = F.pad(a, (0, 0, 0, 1))
            y = self.blk1[i](torch.cat([a, pad[:, self.nb].sum(2)], -1))
            y = y + self.blkg[i](torch.cat([a.mean(1), a.amax(1)], -1)).unsqueeze(1)
            x = x + self.blk2[i](gelu(self.ln2[i](y)))
        return gelu(self.ln_trunk(x))

    def position(self, xpub, tokens):
        projected = self.tok_stem(tokens)
        x = self.trunk(xpub, projected)
        loose = xpub[:, OFF_LOOSE:OFF_LOOSE + LOOSE]
        board = self.board_out(torch.cat([x.mean(1), x.amax(1), loose], -1))
        return board, projected, x

    def board(self, xpub, tokens):
        return self.position(xpub, tokens)[0]

    def configs(self, phi, own, seg):
        counts = phi.reshape(-1, 3, NSLOT).transpose(1, 2)
        u = gelu(self.cfg1(torch.cat([counts, own[seg]], -1))).sum(1)
        u = gelu(self.ln_cfg(u))
        bag = self.cfg_m(own).reshape(-1, NSLOT, 3, POOL)[seg]
        return (self.cfg_f(u),
                self.cfg_g(u) + (counts.unsqueeze(-1) * bag).sum((1, 2)),
                self.cfg_p(u))

    def actions(self, desc, boards, heads, cards, spatial, board_of, head_of):
        row = board_of
        zero_card = cards.new_zeros(cards.shape[0], 1, C)
        card = torch.cat([cards, zero_card], 1)
        coin = desc[:, 1:3].long().clamp_max(NTYPE)
        zero_hex = spatial.new_zeros(spatial.shape[0], 1, C)
        hexes = torch.cat([spatial, zero_hex], 1)
        where = desc[:, 3:6].long().clamp_max(N_HEXES)
        entity = torch.stack([
            card[row, coin[:, 0]], card[row, coin[:, 1]],
            hexes[row, where[:, 0]], hexes[row, where[:, 1]],
            hexes[row, where[:, 2]],
        ], 1)
        present = desc[:, 1:6].lt(255).unsqueeze(-1)
        local = (entity * self.act_role.weight * present).sum(1)
        z = (self.act_kind(desc[:, 0].long()) + local
             + self.act_board(boards)[row] + self.act_h(heads)[head_of])
        return self.act_out(gelu(self.ln_act(z)))

    def join(self, p, pooled, seat):
        z = self.join_p(p) + self.join_b(torch.cat([pooled, seat], -1))
        for i in range(JBLOCKS):
            z = z + self.joinw[i](gelu(self.ln_join[i](z)))
        return self.ln_h(p + self.join_out(gelu(self.ln_jout(z))))

    def heads(self, p, g, weight, seg, nseg):
        pooled = p.new_zeros(nseg, POOL)
        pooled.index_add_(0, seg, g * weight.unsqueeze(1))
        other = torch.arange(nseg, device=seg.device) ^ 1
        pair = torch.cat([pooled, pooled[other]], -1)
        seat = p.new_tensor([-1.0, 1.0]).repeat(p.shape[0]).unsqueeze(1)
        return self.join(p.repeat_interleave(2, 0), pair, seat)


    def evaluate(self, xpub, phi, weight, seg, nseg):
        cards = self.cards(xpub)
        board, projected, spatial = self.position(xpub, self.tokens(xpub, cards))
        own = cards.reshape(-1, 2, NSLOT, TYPE).flatten(0, 1)
        f, g, fp = self.configs(phi, own, seg)
        h = self.heads(board, g, weight, seg, nseg)
        value = (f * h[seg]).sum(1) + self.value_bias
        return value, (cards, board, projected, spatial, fp, h)

    def forward(self, xpub, phi, weight, seg, nseg):
        return self.evaluate(xpub, phi, weight, seg, nseg)[0]


    def flat(self):
        blocks = [m for i in range(BLOCKS)
                  for m in (self.blk1[i], self.blkg[i], self.blk2[i])]
        mats = [
            self.card1, self.card2, self.pile, self.seat,
            self.hex_stem, self.tok_stem, self.pos, self.glob_stem,
            *blocks,
            self.board_out,
            self.cfg1, self.cfg_f, self.cfg_g, self.cfg_m,
            self.cfg_p, self.act_kind, self.act_role, self.act_board, self.act_out,
            self.join_p, self.join_b, *self.joinw, self.join_out,
            self.act_h,
        ]
        norms = [n for i in range(BLOCKS) for n in (self.ln1[i], self.ln2[i])]
        norms += [self.ln_trunk, self.ln_cfg, *self.ln_join,
                  self.ln_jout, self.ln_h, self.ln_act]

        def raw(t):
            return t.detach().cpu().contiguous().numpy().ravel()

        w = [raw(m.weight if isinstance(m, nn.Embedding) else m.weight.t())
             for m in mats]
        b = [raw(m.bias) for m in mats
             if not isinstance(m, nn.Embedding) and m.bias is not None]
        b.append(raw(self.value_bias))
        ln = [x for n in norms for x in (raw(n.weight), raw(n.bias))]
        f = lambda xs: np.ascontiguousarray(np.concatenate(xs), np.float32)
        return f(w), f(b), f(ln)

    def push(self):
        warchest.set_weights(*self.flat())
