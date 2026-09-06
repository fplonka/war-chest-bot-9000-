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
OFF_PILES = warchest.OFF_PILES
OFF_CARDS = warchest.OFF_CARDS
OFF_LOOSE = warchest.OFF_LOOSE
LOOSE = warchest.LOOSE

TYPE = 64
C = 96
BLOCKS = 8
D = 256
ATTN = 128
HEADS = 4
HEAD = ATTN // HEADS
N_KINDS = warchest.N_KINDS


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

        self.cfg_in = nn.Linear(3 + TYPE, ATTN)
        self.cfg_seat = nn.Embedding(2, ATTN)
        self.ln_cfg = nn.LayerNorm(ATTN)
        self.token_in = nn.Linear(C, ATTN)
        self.attn_q = nn.Linear(ATTN, ATTN, bias=False)
        self.attn_k = nn.Linear(ATTN, ATTN, bias=False)
        self.attn_v = nn.Linear(ATTN, ATTN, bias=False)
        self.attn_out = nn.Linear(ATTN, ATTN, bias=False)
        self.ln_attn = nn.LayerNorm(ATTN)

        context = 3 * ATTN + D
        self.value_hidden = nn.Linear(context, ATTN)
        self.value_out = nn.Linear(ATTN, 1)
        self.cfg_policy = nn.Linear(3 * ATTN, D)
        self.head_policy = nn.Linear(2 * ATTN + D, D)

        self.act_kind = nn.Embedding(N_KINDS, C)
        self.act_role = nn.Embedding(5, C)
        self.act_board = nn.Linear(D, C, bias=False)
        self.act_h = nn.Linear(D, C, bias=False)
        self.ln_act = nn.LayerNorm(C)
        self.act_out = nn.Linear(C, D)
        nn.init.normal_(self.act_kind.weight, std=C ** -0.5)
        nn.init.ones_(self.act_role.weight)

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

    def position(self, xpub, cards):
        projected = self.tok_stem(self.tokens(xpub, cards))
        spatial = self.trunk(xpub, projected)
        loose = xpub[:, OFF_LOOSE:OFF_LOOSE + LOOSE]
        board = self.board_out(torch.cat([
            spatial.mean(1), spatial.amax(1), loose], -1))
        return board, projected, spatial

    def configs(self, phi, own, seg):
        counts = phi.reshape(-1, 3, NSLOT).transpose(1, 2)
        query = gelu(self.cfg_in(torch.cat([counts, own[seg]], -1))).sum(1)
        return gelu(self.ln_cfg(query + self.cfg_seat(seg % 2)))

    def attend(self, query, tokens, board_of):
        boards, ntokens = tokens.shape[:2]
        counts = torch.bincount(board_of, minlength=boards)
        starts = counts.cumsum(0) - counts
        order = board_of.argsort()
        slot = torch.empty_like(board_of)
        slot[order] = torch.arange(query.shape[0], device=query.device) - starts[board_of[order]]
        width = int(counts.max())
        q = query.new_zeros(boards, width, ATTN)
        q[board_of, slot] = self.attn_q(query)
        q = q.reshape(boards, width, HEADS, HEAD).transpose(1, 2)
        k = self.attn_k(tokens).reshape(boards, ntokens, HEADS, HEAD).transpose(1, 2)
        v = self.attn_v(tokens).reshape(boards, ntokens, HEADS, HEAD).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(q, k, v)
        mixed = mixed.transpose(1, 2).reshape(boards, width, ATTN)
        mixed = self.attn_out(mixed[board_of, slot])
        return self.ln_attn(query + mixed)

    def belief_context(self, conditioned, weight, seg, nseg):
        pooled = conditioned.new_zeros(nseg, ATTN)
        pooled.index_add_(0, seg, conditioned * weight.unsqueeze(1))
        other = torch.arange(nseg, device=seg.device) ^ 1
        return torch.cat([pooled, pooled[other]], -1)

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

    def evaluate(self, xpub, phi, weight, seg, nseg):
        cards = self.cards(xpub)
        board, projected, spatial = self.position(xpub, cards)
        own = cards.reshape(-1, 2, NSLOT, TYPE).flatten(0, 1)
        query = self.configs(phi, own, seg)
        public = self.token_in(torch.cat([spatial, projected], 1))
        conditioned = self.attend(query, public, seg // 2)
        belief = self.belief_context(conditioned, weight, seg, nseg)
        context = torch.cat([conditioned, board[seg // 2], belief[seg]], -1)
        value = self.value_out(gelu(self.value_hidden(context))).squeeze(1)
        fp = self.cfg_policy(torch.cat([conditioned, belief[seg]], -1))
        head = self.head_policy(torch.cat([
            board.repeat_interleave(2, 0), belief], -1))
        return value, (cards, board, projected, spatial, fp, head)

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
            self.cfg_in, self.cfg_seat, self.token_in,
            self.attn_q, self.attn_k, self.attn_v, self.attn_out,
            self.value_hidden, self.value_out,
            self.cfg_policy, self.head_policy,
            self.act_kind, self.act_role, self.act_board, self.act_out,
            self.act_h,
        ]
        norms = [n for i in range(BLOCKS) for n in (self.ln1[i], self.ln2[i])]
        norms += [self.ln_trunk, self.ln_cfg, self.ln_attn, self.ln_act]

        def raw(t):
            return t.detach().cpu().contiguous().numpy().ravel()

        w = [raw(m.weight if isinstance(m, nn.Embedding) else m.weight.t())
             for m in mats]
        b = [raw(m.bias) for m in mats
             if not isinstance(m, nn.Embedding) and m.bias is not None]
        ln = [x for n in norms for x in (raw(n.weight), raw(n.bias))]
        flatten = lambda xs: np.ascontiguousarray(np.concatenate(xs), np.float32)
        return flatten(w), flatten(b), flatten(ln)

    def push(self):
        warchest.set_weights(*self.flat())
