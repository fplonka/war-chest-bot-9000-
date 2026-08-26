"""The production value network: ``V(PBS, config) -> scalar``.

Shape, and why it is this shape
-------------------------------
Growing-tree CFR calls the network in two different regimes:

* once when a public leaf is created — that physical state never changes;
* once per traversal — the beliefs and queried player do.

The solver therefore caches the board trunk and every config encoding for the
life of the tree. Capacity goes in that amortised path; the repeated
belief-conditioned join stays thin. This is the same split DeepStack and ReBeL
get from a public-state tower. War Chest's config set is variable, so the
network generates one readout row per config instead of using a fixed table.

    physical state ─► TRUNK (hex residual blocks, global pooling) ─► P
                                                                    │  once/leaf
    config c ─────► CONFIG ENCODER ─► f(c) [readout] , g(c) [pool]  │  once/config
                                                                    ▼
 [P, Σβ_own g, Σβ_opp g, seat] ────────────────► JOIN ─► h          every iteration
                                                                    │
                                        v(c) = <f(c), h> + b  ──────┘

The trunk is a KataGo-shaped pre-activation ResNet over the 37 hexes with the
board's own adjacency, plus a global-pooling bias in every block.

Everything that indexes a coin type is permutation-equivariant over the slots:
the ten types are a *set* of tokens described by their printed card facts, so
the same unit reads the same whichever slot the draft put it in, and an unseen
draft is describable rather than an unknown identity.
"""

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

TYPE = 64      # coin-type token width
C = 96         # hex channel width
BLOCKS = 8     # trunk residual blocks
D = 256        # board vector / readout width
POOL = 64      # pooled config embedding width
CFGH = 128     # config encoder hidden width
JW = 128       # join width
JBLOCKS = 3    # join residual blocks
AW = 128       # action encoder hidden width
N_KINDS = warchest.N_KINDS
# What the policy head reads off an action: its kind, the coin slot it spends
# (or none), and the three squares it can name.
AFEAT = N_KINDS + (NSLOT + 1) + 3 * (N_HEXES + 1)

JOIN_IN = 2 * POOL + 1  # both beliefs and the queried physical seat


def gelu(x):
    return F.gelu(x, approximate="tanh")


class Net(nn.Module):
    """A physical-state trunk with player-conditioned value queries."""

    def __init__(self):
        super().__init__()
        # -- coin-type tokens (a set of NTYPE tokens, shared weights) --
        self.card1 = nn.Linear(CARD_FEATS, TYPE)
        self.card2 = nn.Linear(TYPE, TYPE)
        self.pile = nn.Linear(PILE_COUNTS, TYPE, bias=False)
        self.seat = nn.Embedding(2, TYPE)

        # -- trunk stem --
        self.hex_stem = nn.Linear(HEX_FACTS, C)
        self.tok_stem = nn.Linear(TYPE, C, bias=False)
        self.pos = nn.Embedding(N_HEXES, C)
        self.glob_stem = nn.Linear(LOOSE, C, bias=False)

        # -- trunk: pre-activation residual blocks with a global-pooling bias --
        self.blk1 = nn.ModuleList([nn.Linear(2 * C, C) for _ in range(BLOCKS)])
        self.blkg = nn.ModuleList([nn.Linear(2 * C, C) for _ in range(BLOCKS)])
        self.blk2 = nn.ModuleList([nn.Linear(C, C) for _ in range(BLOCKS)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(C) for _ in range(BLOCKS)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(C) for _ in range(BLOCKS)])
        self.ln_trunk = nn.LayerNorm(C)

        self.board_out = nn.Linear(2 * C + LOOSE, D)

        # -- config encoder --
        # Two paths into the pooling vector. The nonlinear one binds a slot's
        # count to its card and can express anything; the linear one is there
        # because pooling happens *after* it, so `Σ_c β(c) g(c)` carries the
        # exact belief-weighted count of every card — "they almost certainly
        # cannot play an Archer this turn" is a marginal, and the join should
        # read it rather than reconstruct it from an average of GELUs.
        self.cfg1 = nn.Linear(3 + TYPE, CFGH)
        self.ln_cfg = nn.LayerNorm(CFGH)
        self.cfg_f = nn.Linear(CFGH, D)
        self.cfg_g = nn.Linear(CFGH, POOL)
        self.cfg_m = nn.Linear(TYPE, 3 * POOL, bias=False)
        # -- policy head --
        # Student of Games' network is value *and policy*. The second readout
        # has the same shape as the first: a config vector dotted with a
        # situation vector, `logit(c, a) = <cfg_p(c), e(a)>`, where `e(a)`
        # describes one action against the board it is played on.
        self.cfg_p = nn.Linear(CFGH, D)
        self.act_in = nn.Linear(AFEAT, AW)
        self.act_board = nn.Linear(D, AW, bias=False)
        self.act_h = nn.Linear(D, AW, bias=False)
        self.ln_act = nn.LayerNorm(AW)
        self.act_out = nn.Linear(AW, D)

        # -- join (the only per-iteration path) --
        self.join_p = nn.Linear(D, JW, bias=False)
        self.join_b = nn.Linear(JOIN_IN, JW)
        self.joinw = nn.ModuleList([nn.Linear(JW, JW) for _ in range(JBLOCKS)])
        self.ln_join = nn.ModuleList([nn.LayerNorm(JW) for _ in range(JBLOCKS)])
        self.ln_jout = nn.LayerNorm(JW)
        self.join_out = nn.Linear(JW, D)
        self.ln_h = nn.LayerNorm(D)
        self.value_bias = nn.Parameter(torch.zeros(1))

        # A dot-product readout has no output matrix to shrink, so the small
        # init lands on the config side: every value starts at the bias.
        nn.init.normal_(self.cfg_f.weight, std=1e-3)
        nn.init.zeros_(self.cfg_f.bias)

        nb = torch.as_tensor(warchest.hex_neighbours(), dtype=torch.long)
        self.register_buffer("nb", nb.view(N_HEXES, 6), persistent=False)
        self.register_buffer("seat_of", torch.arange(NTYPE) // NSLOT,
                             persistent=False)

    # ---------------------------------------------------------------- pieces

    def cards(self, xpub):
        """The printed-card token of each coin type. Fixed for a whole solve."""
        facts = xpub[:, OFF_CARDS:OFF_CARDS + NTYPE * CARD_FEATS]
        facts = facts.reshape(-1, NTYPE, CARD_FEATS)
        return self.card2(gelu(self.card1(facts)))

    def tokens(self, xpub, cards):
        """Card token plus this row's pile counts and the owner's seat."""
        piles = xpub[:, OFF_PILES:OFF_CARDS].reshape(-1, NTYPE, PILE_COUNTS)
        return cards + self.pile(piles) + self.seat(self.seat_of)

    def trunk(self, xpub, tokens):
        """37 physical hex tokens through BLOCKS residual blocks."""
        batch = xpub.shape[0]
        hexes = xpub[:, :N_HEXES * HEX_CH].reshape(batch, N_HEXES, HEX_CH)
        projected = self.tok_stem(tokens)
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

    def board(self, xpub, tokens):
        """One physical board vector."""
        x = self.trunk(xpub, tokens)
        loose = xpub[:, OFF_LOOSE:OFF_LOOSE + LOOSE]
        return self.board_out(torch.cat([x.mean(1), x.amax(1), loose], -1))

    def configs(self, phi, own, seg):
        """Readout vector `f(c)` and pooling vector `g(c)` for each config.

        ``phi`` is ``[n, CCOUNTS]`` normalised hand/face-down/bag counts;
        ``own`` is ``[nseg, NSLOT, TYPE]`` — each query's own card tokens.

        `g` is nonlinear plus linear. The linear half is a count-weighted sum
        of per-zone card embeddings, so `Σ_c β(c) g(c)` contains exactly the
        belief's expected holding of every card, bound to that card.
        """
        counts = phi.reshape(-1, 3, NSLOT).transpose(1, 2)
        u = gelu(self.cfg1(torch.cat([counts, own[seg]], -1))).sum(1)
        u = gelu(self.ln_cfg(u))
        bag = self.cfg_m(own).reshape(-1, NSLOT, 3, POOL)[seg]
        return (self.cfg_f(u),
                self.cfg_g(u) + (counts.unsqueeze(-1) * bag).sum((1, 2)),
                self.cfg_p(u))

    def actions(self, feat, boards, heads, board_of=None, head_of=None):
        """``e(a)`` for actions ``feat`` ``[n, AFEAT]``.

        ``boards`` is ``[rows, D]`` and ``board_of`` says which row each action
        is played on. ``heads`` is the belief-conditioned join output and
        ``head_of`` names the query each action belongs to.
        """
        proj = self.act_board(boards)
        if board_of is not None:
            proj = proj[board_of]
        hproj = self.act_h(heads)
        if head_of is not None:
            hproj = hproj[head_of]
        return self.act_out(gelu(self.ln_act(self.act_in(feat) + proj + hproj)))

    def join(self, p, pooled, seat):
        """The per-iteration path: beliefs and queried seat modulate the board."""
        z = self.join_p(p) + self.join_b(torch.cat([pooled, seat], -1))
        for i in range(JBLOCKS):
            z = z + self.joinw[i](gelu(self.ln_join[i](z)))
        return self.ln_h(p + self.join_out(gelu(self.ln_jout(z))))

    def heads(self, p, g, weight, seg, nseg):
        """Belief-conditioned join rows for all canonical queries."""
        pooled = p.new_zeros(nseg, POOL)
        pooled.index_add_(0, seg, g * weight.unsqueeze(1))
        other = torch.arange(nseg, device=seg.device) ^ 1
        pair = torch.cat([pooled, pooled[other]], -1)
        seat = p.new_tensor([-1.0, 1.0]).repeat(p.shape[0]).unsqueeze(1)
        return self.join(p.repeat_interleave(2, 0), pair, seat)

    # --------------------------------------------------------------- forward

    def forward_parts(self, xpub, phi, weight, seg, nseg):
        """Return values and the features shared by the policy head."""
        cards = self.cards(xpub)
        physical = xpub[0::2]
        board = self.board(physical, self.tokens(physical, cards[0::2]))
        f, g, fp = self.configs(phi, cards[:, :NSLOT], seg)
        heads = self.heads(board, g, weight, seg, nseg)
        value = (f * heads[seg]).sum(1) + self.value_bias
        return value, board, heads, fp

    def forward(self, xpub, phi, weight, seg, nseg):
        """Values for paired physical-state queries.

        Query ``q`` owns configs with ``seg == q``. Queries ``q`` and ``q ^ 1``
        share one physical board trunk and differ by belief order and seat.
        """
        return self.forward_parts(xpub, phi, weight, seg, nseg)[0]

    # ------------------------------------------------------------ weight blob

    def flat(self):
        """The fixed v11 blob read by Rust.

        Order is the contract. Linear matrices are stored ``[in, out]``
        row-major, embeddings ``[n, width]``.
        """
        blocks = [m for i in range(BLOCKS)
                  for m in (self.blk1[i], self.blkg[i], self.blk2[i])]
        mats = [
            self.card1, self.card2, self.pile, self.seat,
            self.hex_stem, self.tok_stem, self.pos, self.glob_stem,
            *blocks,
            self.board_out,
            self.cfg1, self.cfg_f, self.cfg_g, self.cfg_m,
            self.cfg_p, self.act_in, self.act_board, self.act_out,
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
