"""The production value network: ``V(PBS, exact_config) -> scalar``."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest

N_HEXES = warchest.N_HEXES
NTYPE = warchest.NTYPE
NSLOT = warchest.NSLOT
N_UNITS = warchest.N_UNITS
HEX_CH = warchest.HEX_CH
HEX_FACTS = warchest.HEX_FACTS
PILE_COUNTS = warchest.PILE_COUNTS
CARD_FEATS = warchest.CARD_FEATS
OFF_PILES = warchest.OFF_PILES
OFF_CARDS = warchest.OFF_CARDS
OFF_LOOSE = warchest.OFF_LOOSE
LOOSE = warchest.LOOSE

UNIT = 32
SLOT = 64
CONFIG = 128
PUBLIC = 384
CONTEXT = 384
JOINT = 256
PUBLIC_IN = N_HEXES * (HEX_FACTS + UNIT) + NTYPE * (PILE_COUNTS + UNIT) + LOOSE


def gelu(x):
    return F.gelu(x, approximate="tanh")


class Mlp(nn.Module):
    """One fixed architecture shared by both canonical player views."""

    def __init__(self):
        super().__init__()
        self.rule1 = nn.Linear(CARD_FEATS, UNIT)
        self.rule2 = nn.Linear(UNIT, UNIT)
        self.unit_id = nn.Embedding(N_UNITS, UNIT)

        self.public = nn.ModuleList([
            nn.Linear(PUBLIC_IN, PUBLIC),
            nn.Linear(PUBLIC, PUBLIC),
            nn.Linear(PUBLIC, PUBLIC),
        ])
        self.public_norm = nn.ModuleList([nn.LayerNorm(PUBLIC) for _ in self.public])

        self.belief_slot = nn.Linear(3 + UNIT, SLOT)
        self.belief_config = nn.Linear(SLOT, CONFIG)
        self.belief_norm = nn.LayerNorm(CONFIG)
        self.belief_item = nn.Linear(CONFIG, CONFIG)

        self.candidate_slot = nn.Linear(3 + UNIT, SLOT)
        self.candidate = nn.Linear(SLOT, CONFIG)
        self.candidate_norm = nn.LayerNorm(CONFIG)

        self.context = nn.ModuleList([
            nn.Linear(PUBLIC + 2 * CONFIG, CONTEXT),
            nn.Linear(CONTEXT, CONTEXT),
        ])
        self.context_norm = nn.ModuleList([nn.LayerNorm(CONTEXT) for _ in self.context])

        self.context_joint = nn.Linear(CONTEXT, JOINT, bias=False)
        self.candidate_joint = nn.Linear(CONFIG, JOINT, bias=False)
        self.joint_bias = nn.Parameter(torch.zeros(JOINT))
        self.value = nn.Linear(JOINT, 1)
        nn.init.normal_(self.value.weight, std=1e-3)
        nn.init.zeros_(self.value.bias)

    def spec(self):
        return {}

    @property
    def dims(self):
        return [4]

    def units(self, xpub, unit_ids):
        facts = xpub[:, OFF_CARDS:OFF_CARDS + NTYPE * CARD_FEATS]
        facts = facts.reshape(-1, NTYPE, CARD_FEATS)
        return self.rule2(gelu(self.rule1(facts))) + self.unit_id(unit_ids)

    def public_input(self, xpub, units):
        batch = xpub.shape[0]
        hexes = xpub[:, :N_HEXES * HEX_CH].reshape(batch, N_HEXES, HEX_CH)
        occupant = hexes[:, :, HEX_FACTS:] @ units
        piles = xpub[:, OFF_PILES:OFF_CARDS].reshape(batch, NTYPE, PILE_COUNTS)
        return torch.cat([
            hexes[:, :, :HEX_FACTS].reshape(batch, -1),
            occupant.reshape(batch, -1),
            torch.cat([piles, units], -1).reshape(batch, -1),
            xpub[:, OFF_LOOSE:OFF_LOOSE + LOOSE],
        ], -1)

    def public_encode(self, xpub, units):
        x = self.public_input(xpub, units)
        for linear, norm in zip(self.public, self.public_norm):
            x = gelu(norm(linear(x)))
        return x

    @staticmethod
    def _slots(phi, units):
        counts = phi.reshape(-1, 3, NSLOT).transpose(1, 2)
        return torch.cat([counts, units[:, :NSLOT]], -1)

    def belief_configs(self, phi, units):
        x = gelu(self.belief_slot(self._slots(phi, units))).sum(1)
        x = gelu(self.belief_norm(self.belief_config(x)))
        return self.belief_item(x)

    def candidates(self, phi, units):
        x = gelu(self.candidate_slot(self._slots(phi, units))).sum(1)
        return gelu(self.candidate_norm(self.candidate(x)))

    def forward(self, xpub, unit_ids, phi, weight, seg, nseg):
        """Return one value for each config in a ragged canonical-query batch.

        Query ``q`` is public row ``q``. Its own configs have ``seg == q`` and
        its opponent belief is query ``q ^ 1`` from the same physical row.
        """
        units = self.units(xpub, unit_ids)
        public = self.public_encode(xpub, units)
        config_units = units[seg]

        items = self.belief_configs(phi, config_units)
        pooled = torch.zeros(nseg, CONFIG, dtype=items.dtype, device=items.device)
        pooled.index_add_(0, seg, items * weight.unsqueeze(1))
        other = torch.arange(nseg, device=seg.device) ^ 1

        context = torch.cat([public, pooled, pooled[other]], -1)
        for linear, norm in zip(self.context, self.context_norm):
            context = gelu(norm(linear(context)))
        common = self.context_joint(context)

        candidate = self.candidate_joint(self.candidates(phi, config_units))
        joint = gelu(common[seg] + candidate + self.joint_bias)
        return self.value(joint).squeeze(1)

    def flat(self):
        """The fixed v4 blob read by Rust and CUDA."""
        matrices = [
            self.rule1, self.rule2,
            *self.public,
            self.belief_slot, self.belief_config, self.belief_item,
            self.candidate_slot, self.candidate,
            *self.context,
            self.context_joint, self.candidate_joint, self.value,
        ]
        weights = [m.weight.detach().cpu().t().contiguous().numpy().ravel() for m in matrices]
        weights.insert(2, self.unit_id.weight.detach().cpu().contiguous().numpy().ravel())
        biases = [m.bias.detach().cpu().numpy().ravel() for m in matrices if m.bias is not None]
        biases.insert(-1, self.joint_bias.detach().cpu().numpy().ravel())
        norms = []
        for n in [*self.public_norm, self.belief_norm, self.candidate_norm, *self.context_norm]:
            norms.extend([n.weight.detach().cpu().numpy().ravel(),
                          n.bias.detach().cpu().numpy().ravel()])
        f = lambda xs: np.ascontiguousarray(np.concatenate(xs), np.float32)
        return f(weights), f(biases), f(norms)

    def push(self, slot):
        warchest.set_weights(self.dims, *self.flat(), slot)
