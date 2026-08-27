"""Student of Games training for War Chest.

Self-play where every decision grows a search tree along trajectories sampled
from its changing strategy. The root of each solve becomes a value target --
one value per config in each player's belief -- and the leaves it queried
become roots for solves of their own, which is how the value function becomes
accurate away from the line of play.

There is a short greedy warm start. A game that reaches the play cap is a draw
with zero utility.

Generation runs in Rust across all CPU cores while the previous batch trains
on the GPU. Python publishes weights between generation batches.

A training row is a public state plus, for each player, the whole belief: the
exact configs in support, their probabilities, and the value the solve gave
each. The config lists are ragged, so they live in a flat arena and a batch is
assembled by gathering spans -- see `Buffer`.

The run snapshots every `snapshot_every` minutes, including the live replay
buffer so a resumed run continues training immediately. Training starts with a
short greedy warm phase labelled by the public static evaluation, then the SoG
phase. When training ends, the snapshots are packed as bots for evaluation by
`tools/arena.py`.

    python train/train.py out=seat
    python train/train.py out=seat note="centre the seat bit at +-0.5"
    python train/train.py out=smoke minutes=6
"""

import argparse
import collections
import dataclasses
import json
import math
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest
import config
import mirror
from export_weights import load as load_checkpoint
from value_net import AFEAT, Net

ROOT = pathlib.Path(__file__).resolve().parent.parent

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
ROW_BYTES = warchest.ROW_BYTES
ACT_BYTES = warchest.ACT_BYTES
N_KINDS = warchest.N_KINDS
NSLOT = warchest.NSLOT
N_HEXES = warchest.N_HEXES

# What `SolveFarm.collect` reports about the device rounds, cumulative
# since the farm started. `rounds` is the denominator of the other three.
ROUND_KEYS = ("rounds", "round_calls", "round_rows", "round_nanos", "budget_hits")


def scheduled_lr(initial, final, elapsed, duration, stable_frac):
    """Flat warm self-play rate followed by a cosine decay to ``final``."""
    stable = stable_frac * duration
    if stable_frac >= 1.0 or elapsed <= stable:
        return initial
    if elapsed >= duration:
        return final
    return final + (initial - final) * (1.0 + math.cos(
        math.pi * ((elapsed - stable) / (duration - stable)))) / 2.0


def action_feats(pa):
    """The five stored bytes of each action into the head's one-hot input.

    The layout is the contract with `Net::action_feats`: kind, the coin slot it
    spends as the one-hot column (`NSLOT` meaning none, already so in the row),
    then the three squares it names with the last column meaning no square.
    """
    feat = np.zeros((len(pa), AFEAT), np.float32)
    if not len(pa):
        return feat
    idx = np.arange(len(pa))
    feat[idx, pa[:, 0]] = 1.0
    feat[idx, N_KINDS + pa[:, 1]] = 1.0
    at = N_KINDS + NSLOT + 1
    for k in range(3):
        h = np.where(pa[:, 2 + k] == 255, N_HEXES, pa[:, 2 + k].astype(np.int64))
        feat[idx, at + h] = 1.0
        at += N_HEXES + 1
    return feat


def expand_batch(rows):
    """Expand packed replay rows into the public encoding, in one batch.
    The expansion itself runs in Rust — one source of truth with the
    solver's leaf encoding."""
    n = len(rows)
    return np.asarray(warchest.expand_rows(rows.ravel()), np.float32).reshape(n, -1)


class Buffer:
    """Fixed-capacity FIFO over rows whose config lists are ragged.

    One FIFO of rows. Each row occupies a slot in the row ring and a span in
    the config, action and policy-cell rings. Append retires the oldest rows
    until the new chunk fits every ring, then writes. No ring is walked on its
    own: a row is live exactly while `lo` still names it.

    Bootstrapped targets are averaged over whatever history the buffer holds, so
    its length is a real algorithmic knob and not just a memory setting -- the
    reference implementation runs a 2M buffer. A row is the frozen compact
    format (`ROW_BYTES` raw bytes: hex facts, piles, unit ids, and scalars) --
    ~223 bytes instead of the ~1.9 KB the old float encoding cost -- and the
    network input is expanded from it when a batch is made. Counts are stored
    as the `uint8` they are and everything else as float16, which is what makes
    the cap affordable: a row costs `ROW_BYTES` bytes plus 20 per config.

    Preallocated and written with wraparound rather than grown by
    concatenation. The concatenate form rebuilt the whole buffer every epoch:
    at an 800k cap that copies ~2.6 GB per epoch and transiently holds two
    copies of it, which is most of a 16 GB machine. `np.zeros` maps zero pages
    lazily, so reserving the full capacity up front costs nothing until it is
    actually filled.

    Solve offsets are kept in row space (`soff`), so a dump can be split at
    solve boundaries for honest offline comparisons.
    """

    def __init__(self, cap, ccap):
        self.cap, self.ccap = cap, ccap
        self.x = np.zeros((cap, ROW_BYTES), np.uint8)
        self.soff = np.zeros(0, np.int64)
        self.cstart = np.zeros(cap, np.int64)   # absolute arena offset
        self.clen = np.zeros((cap, 2), np.int32)
        self.cc = np.zeros((ccap, CCOUNTS), np.uint8)
        self.cp = np.zeros(ccap, np.uint8)
        self.cw = np.zeros(ccap, np.float16)
        self.cy = np.zeros(ccap, np.float16)
        # The policy target, per row: the root's actions, and the legal cells
        # with their probability. Only main-line rows carry one, so both arenas
        # are sized off the row cap rather than the config cap.
        self.pastart = np.zeros(cap, np.int64)
        self.palen = np.zeros(cap, np.int32)
        self.pcstart = np.zeros(cap, np.int64)
        self.pclen = np.zeros(cap, np.int32)
        self.acap = cap * 24
        self.pcap = cap * 96
        self.pa = np.zeros((self.acap, ACT_BYTES), np.uint8)
        self.pci = np.zeros(self.pcap, np.uint16)
        self.pact = np.zeros(self.pcap, np.uint16)
        self.pp = np.zeros(self.pcap, np.float16)
        self.written_at = np.zeros(cap, np.float64)
        self.created_at = np.zeros(cap, np.float64)
        self.source = np.zeros(cap, np.uint8)  # 0 warm, 1 play, 2 query
        self.truth = np.zeros((cap, 2), np.uint32)
        self.outcome = np.full((cap, 2), np.nan, np.float32)
        self.td1 = np.zeros(cap, np.uint8)
        self.acts = 0
        self.cells = 0
        self.rows = 0   # rows ever written
        self.cfgs = 0   # configs ever written
        self.lo = 0     # oldest row whose configs are still in the arena

    _ROW_FIELDS = (
        "x", "cstart", "clen", "pastart", "palen", "pcstart", "pclen",
        "written_at", "created_at", "source", "truth", "outcome", "td1")
    _CONFIG_FIELDS = ("cc", "cp", "cw", "cy")
    _ACTION_FIELDS = ("pa",)
    _CELL_FIELDS = ("pci", "pact", "pp")

    @staticmethod
    def _compact_arena(starts, lens, total, size):
        used = lens > 0
        base = int(starts[used].min()) if np.any(used) else total
        return base, starts - base, (
            np.arange(base, total, dtype=np.int64) % size)

    def state_dict(self):
        """Return the live replay state in compact row and arena order."""
        ids = np.arange(self.lo, self.rows, dtype=np.int64)
        ring = ids % self.cap
        state = {name: getattr(self, name)[ring].copy()
                 for name in self._ROW_FIELDS}
        for start, starts, lens, total, size, fields in (
                ("cstart", self.cstart[ring], self.clen[ring].sum(1),
                 self.cfgs, self.ccap, self._CONFIG_FIELDS),
                ("pastart", self.pastart[ring], self.palen[ring],
                 self.acts, self.acap, self._ACTION_FIELDS),
                ("pcstart", self.pcstart[ring], self.pclen[ring],
                 self.cells, self.pcap, self._CELL_FIELDS)):
            base, offsets, at = self._compact_arena(starts, lens, total, size)
            state[start] = offsets.copy()
            state.update({name: getattr(self, name)[at].copy()
                          for name in fields})
        state.update({
            "soff": self.soff.copy(),
            "acts": int(self.acts),
            "cells": int(self.cells),
            "rows": int(self.rows),
            "cfgs": int(self.cfgs),
            "lo": int(self.lo),
        })
        return state

    def load_state_dict(self, state):
        """Restore a state_dict into this buffer."""
        lo, rows = int(state["lo"]), int(state["rows"])
        if lo < 0 or rows < lo or rows - lo > self.cap:
            raise ValueError(f"invalid replay row range {lo}:{rows}")
        self.__init__(self.cap, self.ccap)
        ids = np.arange(lo, rows, dtype=np.int64)
        ring = ids % self.cap
        self.lo, self.rows = lo, rows
        self.cfgs, self.acts, self.cells = (
            int(state["cfgs"]), int(state["acts"]), int(state["cells"]))
        for name in self._ROW_FIELDS:
            getattr(self, name)[ring] = state[name]
        for start, fields, total, size in (
                ("cstart", self._CONFIG_FIELDS, self.cfgs, self.ccap),
                ("pastart", self._ACTION_FIELDS, self.acts, self.acap),
                ("pcstart", self._CELL_FIELDS, self.cells, self.pcap)):
            base = total - len(state[fields[0]])
            at = (np.arange(len(state[fields[0]]), dtype=np.int64) + base) % size
            for name in fields:
                getattr(self, name)[at] = state[name]
            getattr(self, start)[ring] += base
        self.soff = np.asarray(state["soff"], np.int64).copy()

    def add(self, x, cc, cw, cy, coff, soff, source, truth, outcome, created,
            td1, pol=None):
        n = len(x)
        lens = np.diff(coff).reshape(n, 2)
        m = len(cw)
        if pol is None:
            na = nc = 0
        else:
            pa, paoff, pcoff, pci, pact, pprob = pol
            na, nc = len(pa), len(pci)
        # Retire the oldest rows until this chunk fits every ring.
        while self.lo < self.rows:
            r = self.lo % self.cap
            if (self.rows - self.lo + n <= self.cap
                    and self.cfgs - self.cstart[r] + m <= self.ccap
                    and self.cells - self.pcstart[r] + nc <= self.pcap
                    and self.acts - self.pastart[r] + na <= self.acap):
                break
            self.lo += 1
        cp = np.repeat(np.tile([0, 1], n).astype(np.uint8), lens.ravel())
        starts = self.cfgs + coff[:-1:2]
        base = self.rows
        now = time.time()
        for i in range(0, n, 4096):
            j = min(i + 4096, n)
            sl = np.arange(i, j) + base
            self.x[sl % self.cap] = x[i:j]
            self.cstart[sl % self.cap] = starts[i:j]
            ring = sl % self.cap
            self.clen[ring] = lens[i:j]
            self.written_at[ring] = now
            self.created_at[ring] = created[i:j]
            self.source[ring] = source[i:j]
            self.truth[ring] = truth[i:j]
            self.outcome[ring] = outcome[i:j]
            self.td1[ring] = td1[i:j]
        sl = (np.arange(m) + self.cfgs) % self.ccap
        self.cc[sl], self.cp[sl], self.cw[sl], self.cy[sl] = cc, cp, cw, cy
        if pol is not None:
            alen = np.diff(paoff).astype(np.int32)
            clen = np.diff(pcoff).astype(np.int32)
            for i in range(0, n, 4096):
                j = min(i + 4096, n)
                sl = (np.arange(i, j) + base) % self.cap
                self.pastart[sl] = self.acts + paoff[i:j]
                self.palen[sl] = alen[i:j]
                self.pcstart[sl] = self.cells + pcoff[i:j]
                self.pclen[sl] = clen[i:j]
            self.pa[(np.arange(na) + self.acts) % self.acap] = pa
            at = (np.arange(nc) + self.cells) % self.pcap
            self.pci[at], self.pact[at], self.pp[at] = pci, pact, pprob
            self.acts += na
            self.cells += nc
        self.rows += n
        self.cfgs += m
        # Solve offsets in absolute row space (first entry 0, trailing count).
        self.soff = np.concatenate([self.soff, np.asarray(soff, np.int64)[1:] + base])
        # Drop offsets whose rows have been evicted. Without this, soff grows
        # for the whole run and the concatenate above copies all of it every
        # chunk — quadratic in the run length.
        if self.soff.size:
            i = int(np.searchsorted(self.soff, self.lo, "right"))
            if i > self.soff.size // 2:
                self.soff = self.soff[i:].copy()

    def span_seconds(self):
        return (time.time() - self.written_at[self.lo % self.cap]
                if self.lo < self.rows else 0.0)

    def clear(self):
        self.lo = self.rows
        self.soff = np.zeros(0, np.int64)

    def __len__(self):
        return self.rows - self.lo

    def gather(self, ids):
        """Assemble a batch from absolute row ids.

        Returns `(rows, cc, cp, cw, cy, seg)`.
        """
        s = ids % self.cap
        lens = self.clen[s].sum(1).astype(np.int64)
        total = int(lens.sum())
        # Arena indices of every config of every chosen row, flattened.
        base = np.repeat(self.cstart[s], lens)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        at = (base + within) % self.ccap
        seg = 2 * np.repeat(np.arange(len(ids), dtype=np.int64), lens) + self.cp[at]
        # The policy target, remapped onto the batch. A row with no target
        # contributes no cells, which is how a query solve drops out of the
        # policy loss without a mask.
        alen, clen = self.palen[s].astype(np.int64), self.pclen[s].astype(np.int64)
        ai = (np.repeat(self.pastart[s], alen)
              + (np.arange(int(alen.sum()), dtype=np.int64)
                 - np.repeat(np.concatenate([[0], np.cumsum(alen)[:-1]]), alen)))
        ci = (np.repeat(self.pcstart[s], clen)
              + (np.arange(int(clen.sum()), dtype=np.int64)
                 - np.repeat(np.concatenate([[0], np.cumsum(clen)[:-1]]), clen)))
        # An action's index becomes batch-global, and a cell names the query
        # (row, acting config) it belongs to.
        abase = np.concatenate([[0], np.cumsum(alen)[:-1]])
        cellrow = np.repeat(np.arange(len(ids), dtype=np.int64), clen)
        # A cell names its config within its own row; the batch arena puts that
        # row's configs at `rowbase`, so the two add to an arena index.
        rowbase = np.concatenate([[0], np.cumsum(lens)[:-1]])
        pcfg = np.repeat(rowbase, clen) + self.pci[ci % self.pcap].astype(np.int64)
        pp = self.pp[ci % self.pcap].astype(np.float32)
        pol = (self.pa[ai % self.acap],
               np.repeat(abase, clen) + self.pact[ci % self.pcap],
               cellrow, pcfg, pp,
               np.repeat(np.arange(len(ids), dtype=np.int64), alen))
        cw = self.cw[at].astype(np.float32)
        mass = np.bincount(seg, weights=cw, minlength=2 * len(ids)).astype(np.float32)
        cw /= mass[seg]
        return (self.x[s], self.cc[at], self.cp[at], cw,
                self.cy[at].astype(np.float32), seg, pol)

    def sample_ids(self, batch, rng, recent_mix=0.0, recent_frac=0.2):
        """Row ids, with part drawn from the newest rows only.

        Uniform sampling over a 2M-row buffer is not neutral here: the targets
        are bootstrapped, so an old row's target was written by an old network
        and is wrong by however much the network has moved since. Sampling
        purely by recency is not the fix either, because held-out error falls
        monotonically with the number of *distinct* positions trained on
        (40k -> 0.0122, 284k -> 0.0082), and a recency-only sampler throws that
        away to refit a small window.

        So: a mixture. `recent_mix` of the batch comes from the newest
        `recent_frac` of the buffer and the rest from all of it, which draws a
        row in the fresh slice `1 + mix / ((1 - mix) * frac)` times as often as
        an old one -- 6x at the defaults, or 3x the average rate -- while
        leaving every row reachable. Two uniform draws, and no weight vector to
        rebuild each epoch.

        """
        ids = rng.integers(self.lo, self.rows, size=batch)
        k = int(batch * recent_mix)
        if k > 0:
            span = max(1, int((self.rows - self.lo) * recent_frac))
            ids[:k] = rng.integers(self.rows - span, self.rows, size=k)
        return ids

    def sample(self, batch, rng, recent_mix=0.0, recent_frac=0.2):
        return self.gather(self.sample_ids(batch, rng, recent_mix, recent_frac))

    def sample_old(self, batch, rng, recent_frac=0.2):
        """A batch from outside the recent slice — the stale majority — for
        the age-bucket loss. A diagnostic, not training."""
        span = max(1, int((self.rows - self.lo) * recent_frac))
        hi = max(self.lo + 1, self.rows - span)
        return self.gather(rng.integers(self.lo, hi, size=batch))

    def replay_stats(self):
        ids = np.arange(self.lo, self.rows, dtype=np.int64) % self.cap
        n = max(len(ids), 1)
        source = np.bincount(self.source[ids], minlength=3)
        configs = self.clen[ids].sum(dtype=np.int64)
        return {
            "replay_warm_frac": source[0] / n,
            "replay_play_frac": source[1] / n,
            "replay_query_frac": source[2] / n,
            "replay_td1_row_frac": float(self.td1[ids].sum()) / n,
            "replay_td1_target_frac": 2.0 * self.td1[ids].sum() / max(configs, 1),
            "target_age_max": (time.time() - self.created_at[ids].min()
                               if len(ids) else 0.0),
        }

    def sample_calibration(self, batch, rng):
        """Rows with game outcomes, plus their true-config indices."""
        ids = np.arange(self.lo, self.rows, dtype=np.int64)
        ids = ids[np.isfinite(self.outcome[ids % self.cap, 0])]
        if not len(ids):
            return None
        ids = rng.choice(ids, size=min(batch, len(ids)), replace=False)
        ring = ids % self.cap
        parts = self.gather(ids)
        lens = self.clen[ring].astype(np.int64)
        start = np.concatenate([[0], np.cumsum(lens.sum(1))[:-1]])
        at = np.empty(2 * len(ids), np.int64)
        at[0::2] = start + self.truth[ring, 0]
        at[1::2] = start + lens[:, 0] + self.truth[ring, 1]
        return parts, at, self.outcome[ring].ravel()

    def ordered(self):
        """Live rows oldest-first.

        Age order is what makes an honest held-out split possible: rows from
        one epoch come from the same games and are heavily correlated, so a
        random split leaks. Splitting by recency does not.
        """
        return self.gather(np.arange(self.lo, self.rows))


def make_batch(parts, rng, device):
    """Numpy replay batch -> the two canonical player queries per row."""
    del rng
    rows, cc, cp, cw, cy, seg, pol = parts
    n = len(rows)
    views = np.empty((2 * n, ROW_BYTES), np.uint8)
    views[0::2] = rows
    views[1::2] = mirror.mirror_rows(rows)
    x = expand_batch(views)
    phi = cc.astype(np.float32) / CNORM
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)
    pa, pact, pcrow, pcfg, pprob, parow = pol
    policy = (t(action_feats(pa)), t(parow, torch.long), t(pact, torch.long),
              t(pcrow, torch.long), t(pcfg, torch.long), t(pprob))
    return (t(x), t(phi), t(cw), t(seg, torch.long), t(cy), 2 * n, policy)


def forward_values(net, parts):
    # `nseg` is the sixth element, not the last one: a batch carries the policy
    # target after it.
    return net(*parts[:4], parts[5])


def losses(net, xpub, phi, w, seg, y, nseg, policy=None, wp=1.0, stats=None):
    """Value Huber, mean per belief support and then across queries, plus the
    policy cross-entropy Student of Games trains the second head with."""
    v = net(xpub, phi, w, seg, nseg)
    if stats is not None:
        expected = torch.zeros(nseg, dtype=v.dtype, device=v.device)
        expected.index_add_(0, seg, v.detach() * w)
        residual = expected[0::2] + expected[1::2]
        maximum, square_sum = torch.stack([
            residual.abs().max(), residual.square().sum()]).cpu().tolist()
        stats["zero_sum_max"] = max(stats["zero_sum_max"], maximum)
        stats["zero_sum_square_sum"] += square_sum
        stats["zero_sum_n"] += len(residual)
    per = F.smooth_l1_loss(v, y, reduction="none", beta=0.5)
    total = torch.zeros(nseg, dtype=per.dtype, device=per.device)
    count = torch.zeros(nseg, dtype=per.dtype, device=per.device)
    total.index_add_(0, seg, per)
    count.index_add_(0, seg, torch.ones_like(per))
    loss = (total / count.clamp(min=1)).mean()
    # `L` and `L/var` are the *value* loss, as they were before the policy head
    # existed, so the run report still compares with every run before it. The
    # policy term is reported beside them, never folded into them.
    if stats is not None:
        stats["value_loss"] = float(loss.detach())
    if policy is not None and wp > 0.0:
        pl = policy_loss(net, xpub, phi, w, seg, nseg, policy, stats)
        if pl is not None:
            loss = loss + wp * pl
    return loss


def policy_loss(net, xpub, phi, weight, seg, nseg, policy, stats=None):
    """Cross entropy of the policy head against the search's root average.

    The head scores a `(config, action)` cell as `<f_p(c), e(a)>`, so the batch
    is exactly the cells the solves stored. Each cell's softmax runs over its
    own `(row, config)` group, which is one information state.
    """
    feat, parow, pact, _pcrow, pcfg, target = policy
    if feat.shape[0] == 0 or pact.shape[0] == 0:
        return None
    cards = net.cards(xpub)
    physical = xpub[0::2]
    board = net.board(physical, net.tokens(physical, cards[0::2]))
    _f, g, fp = net.configs(phi, cards[:, :NSLOT], seg)
    h = net.heads(board, g, weight, seg, nseg)
    action_query = torch.zeros(feat.shape[0], dtype=torch.long, device=feat.device)
    action_query.scatter_(0, pact, seg[pcfg])
    e = net.actions(feat, board, h, parow, action_query)

    # `pcfg` is already an index into the batch's own config arena, so the cell
    # reads its config vector directly.
    logit = (fp[pcfg] * e[pact]).sum(1)

    # One softmax per information state, over its own cells. `pcfg` is already
    # unique across the batch, so it alone names the group.
    group = pcfg
    uniq, inv = torch.unique(group, return_inverse=True)
    top = torch.full((len(uniq),), -1e30, device=logit.device)
    top = top.scatter_reduce(0, inv, logit, reduce="amax")
    ex = (logit - top[inv]).exp()
    tot = torch.zeros(len(uniq), device=ex.device).index_add_(0, inv, ex)
    logp = (logit - top[inv]) - tot[inv].clamp(min=1e-30).log()
    per = -(target * logp)
    out = torch.zeros(len(uniq), device=per.device).index_add_(0, inv, per)
    loss = out.mean()
    if stats is not None:
        target_mass = torch.zeros(len(uniq), device=target.device).index_add_(
            0, inv, target)
        q = target / target_mass[inv].clamp(min=1e-30)
        target_entropy = torch.zeros(len(uniq), device=target.device).index_add_(
            0, inv, -(q * q.clamp(min=1e-30).log()))
        prior = logp.exp()
        prior_entropy = torch.zeros(len(uniq), device=target.device).index_add_(
            0, inv, -(prior * logp))
        search_ce = torch.zeros(len(uniq), device=target.device).index_add_(
            0, inv, -(q * logp))
        values = torch.stack([
            loss, target_entropy.mean(), prior_entropy.mean(),
            (search_ce - target_entropy).mean()]).detach().cpu().tolist()
        stats.update(dict(zip((
            "policy_loss", "policy_target_entropy", "policy_prior_entropy",
            "policy_search_kl"), values)))
        stats["policy_groups"] = len(uniq)
    return loss


@torch.no_grad()
def diagnostics(net, buf, probe, batch, rng, device, batch_fn, recent_frac):
    """Held replay error, prediction spread, and outcome calibration."""
    nan = float("nan")
    out = {
        "probe_std": float(forward_values(net, probe).std()) if probe is not None else nan,
        "loss_old": nan,
        "loss_new": nan,
        "value_outcome_rmse": nan,
        "value_outcome_mae": nan,
        "value_outcome_bias": nan,
        "value_outcome_corr": nan,
        "value_calibration_slope": nan,
    }
    if len(buf) < batch:
        return out
    old = batch_fn(buf.sample_old(batch, rng, recent_frac), rng, device)
    new = batch_fn(buf.sample(batch, rng, recent_mix=1.0, recent_frac=recent_frac),
                   rng, device)
    out["loss_old"] = float(losses(net, *old, wp=0.0))
    out["loss_new"] = float(losses(net, *new, wp=0.0))
    calibration = buf.sample_calibration(batch, rng)
    if calibration is None:
        return out
    sampled, at, outcome = calibration
    parts = batch_fn(sampled, rng, device)
    pred = forward_values(net, parts)[torch.as_tensor(at, device=device)].float().cpu().numpy()
    error = pred - outcome
    pc = pred - pred.mean()
    oc = outcome - outcome.mean()
    cov = float(np.mean(pc * oc))
    out.update({
        "value_outcome_rmse": float(np.sqrt(np.mean(error * error))),
        "value_outcome_mae": float(np.mean(np.abs(error))),
        "value_outcome_bias": float(error.mean()),
        "value_outcome_corr": cov / max(float(pc.std() * oc.std()), 1e-12),
        "value_calibration_slope": cov / max(float(np.mean(pc * pc)), 1e-12),
    })
    return out


def train_steps(net, opt, buf, steps, batch, rng, device,
                recent_mix=0.0, recent_frac=0.2, profile_cuda=False,
                batch_fn=make_batch, policy_w=0.0, deadline=None):
    """Mean loss over up to `steps` Adam updates -- value, plus the policy
    head's cross entropy at weight `policy_w` -- stopping between updates at
    `deadline`."""
    policy_metrics = (
        "policy_loss", "policy_target_entropy", "policy_prior_entropy",
        "policy_search_kl")
    stat = {"sample_s": 0.0, "prepare_s": 0.0, "forward_wall_s": 0.0,
            "backward_wall_s": 0.0, "batch_configs": 0, "steps": 0,
            "gpu_forward_s": 0.0, "gpu_backward_s": 0.0,
            "zero_sum_max": 0.0, "zero_sum_square_sum": 0.0,
            "zero_sum_n": 0, "grad_clipped": 0,
            "grad_norm_sum": 0.0, "grad_norm_max": 0.0,
            "policy_steps": 0, "sample_ages": [], "sample_delays": [],
            "sample_warm": 0, "sample_play": 0, "sample_query": 0,
            "sample_warm_delay_sum": 0.0, "sample_play_delay_sum": 0.0,
            "sample_query_delay_sum": 0.0,
            "sample_td1_targets": 0, "sample_targets": 0,
            **{f"{key}_sum": 0.0 for key in policy_metrics}}
    if len(buf) < batch:
        return float("nan"), stat
    tot = 0.0
    event_pairs = []
    stream = torch.cuda.current_stream(device) if profile_cuda and device.type == "cuda" else None
    for _ in range(steps):
        if deadline is not None and time.time() >= deadline:
            break
        ts = time.perf_counter()
        ids = buf.sample_ids(batch, rng, recent_mix, recent_frac)
        ring = ids % buf.cap
        stat["sample_ages"].append(time.time() - buf.created_at[ring])
        delay = buf.written_at[ring] - buf.created_at[ring]
        stat["sample_delays"].append(delay)
        source_id = buf.source[ring]
        source = np.bincount(source_id, minlength=3)
        stat["sample_warm"] += int(source[0])
        stat["sample_play"] += int(source[1])
        stat["sample_query"] += int(source[2])
        for source_id_value, name in enumerate(("warm", "play", "query")):
            stat[f"sample_{name}_delay_sum"] += float(
                delay[source_id == source_id_value].sum())
        stat["sample_td1_targets"] += 2 * int(buf.td1[ring].sum())
        stat["sample_targets"] += int(buf.clen[ring].sum())
        sampled = buf.gather(ids)
        stat["sample_s"] += time.perf_counter() - ts
        stat["batch_configs"] += len(sampled[1])
        ts = time.perf_counter()
        parts = batch_fn(sampled, rng, device)
        stat["prepare_s"] += time.perf_counter() - ts
        if stream is not None:
            f0 = torch.cuda.Event(enable_timing=True)
            f1 = torch.cuda.Event(enable_timing=True)
            b1 = torch.cuda.Event(enable_timing=True)
            f0.record(stream)
        ts = time.perf_counter()
        step_stat = {"zero_sum_max": 0.0, "zero_sum_square_sum": 0.0,
                     "zero_sum_n": 0}
        value = losses(net, *parts, wp=policy_w, stats=step_stat)
        tot += step_stat["value_loss"]
        stat["zero_sum_max"] = max(stat["zero_sum_max"], step_stat["zero_sum_max"])
        stat["zero_sum_square_sum"] += step_stat["zero_sum_square_sum"]
        stat["zero_sum_n"] += step_stat["zero_sum_n"]
        if "policy_loss" in step_stat:
            stat["policy_steps"] += 1
            for key in policy_metrics:
                stat[f"{key}_sum"] += step_stat[key]
        stat["forward_wall_s"] += time.perf_counter() - ts
        if stream is not None:
            f1.record(stream)
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        value.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(net.parameters(), 5.0))
        stat["grad_norm_sum"] += grad_norm
        stat["grad_norm_max"] = max(stat["grad_norm_max"], grad_norm)
        stat["grad_clipped"] += int(grad_norm > 5.0)
        opt.step()
        stat["steps"] += 1
        stat["backward_wall_s"] += time.perf_counter() - ts
        if stream is not None:
            b1.record(stream)
            event_pairs.append((f0, f1, b1))
    if event_pairs:
        torch.cuda.synchronize(device)
        stat["gpu_forward_s"] = sum(a.elapsed_time(b) for a, b, _ in event_pairs) / 1000.0
        stat["gpu_backward_s"] = sum(b.elapsed_time(c) for _, b, c in event_pairs) / 1000.0
    return tot / stat["steps"] if stat["steps"] else float("nan"), stat


def ingest(buf, data, warm=False):
    """Append one `gen_data` / farm collect dict onto the replay buffer."""
    x = np.asarray(data["rows"], np.uint8).reshape(-1, ROW_BYTES)
    cc = np.asarray(data["cc"], np.uint8).reshape(-1, CCOUNTS)
    cw = np.asarray(data["cw"], np.float32)
    cy = np.asarray(data["cy"], np.float32)
    bad_cw = int((~np.isfinite(cw)).sum())
    bad_cy = int((~np.isfinite(cy)).sum())
    if bad_cw or bad_cy:
        raise SystemExit(
            f"non-finite collect values: data['cw']={bad_cw}, data['cy']={bad_cy}")
    if not len(x):
        return 0
    cy = np.clip(cy, -1.0, 1.0)
    coff = np.asarray(data["coff"], np.int64)
    soff = np.asarray(data["soff"], np.int64)
    query = np.asarray(data["query"], np.uint8)
    source = np.where(query != 0, 2, 0 if warm else 1).astype(np.uint8)
    truth = np.asarray(data["truth"], np.uint32).reshape(-1, 2)
    outcome = np.asarray(data["outcome"], np.float32).reshape(-1, 2)
    created = np.asarray(data["created"], np.float64)
    td1 = np.asarray(data["td1"], np.uint8)
    pol = (np.asarray(data["pa"], np.uint8).reshape(-1, ACT_BYTES),
           np.asarray(data["paoff"], np.int64),
           np.asarray(data["pcoff"], np.int64),
           np.asarray(data["pci"], np.uint16),
           np.asarray(data["pcell"], np.uint16),
           np.asarray(data["pprob"], np.float16))
    buf.add(x, cc, cw.astype(np.float16), cy.astype(np.float16), coff, soff,
            source, truth, outcome, created, td1, pol)
    return len(x)


def physical_cpus():
    """Return one Linux hardware thread from each physical core."""
    cpus = set()
    root = "/sys/devices/system/cpu"
    if not os.path.isdir(root):
        return []
    for name in os.listdir(root):
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        path = os.path.join(root, name, "topology", "thread_siblings_list")
        try:
            first = open(path).read().strip().split(",", 1)[0].split("-", 1)[0]
            cpus.add(int(first))
        except (OSError, ValueError):
            pass
    return sorted(cpus)


def append_epoch(args, rec):
    with open(f"{args.out}/epochs.jsonl", "a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def last_epoch(args):
    try:
        with open(f"{args.out}/epochs.jsonl", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - (64 << 10)))
            lines = f.read().splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        try:
            return int(json.loads(line)["epoch"]) + 1
        except (KeyError, TypeError, UnicodeDecodeError, ValueError):
            pass
    return 0


def write_log(args, snaps):
    """Write settings and the snapshot manifest atomically."""
    path = f"{args.out}/log.json"
    tmp = path + ".tmp"
    cfg = dataclasses.asdict(args)
    cfg["resume"] = ""
    with open(tmp, "w") as f:
        json.dump({"cfg": cfg, "snapshots": snaps}, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def cpu_state(net):
    return {name: value.detach().cpu().clone()
            for name, value in net.state_dict().items()}


def publish_state(state):
    net = Net()
    net.load_state_dict(state)
    net.push()


def main():
    ap = argparse.ArgumentParser(
        description="Train one run and pack its snapshots as bots.")
    ap.add_argument("over", nargs="*", help="knob=value (production defaults)")
    over = config.parse(ap.parse_args().over)
    resume = over.pop("resume", "")
    name = over.pop("out", None)
    checkpoint = None
    if resume:
        minutes = over.pop("minutes", None)
        if over:
            raise SystemExit("resume only accepts a minutes extension")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        args = config.Cfg(**checkpoint["cfg"])
        if minutes is not None:
            if minutes < args.minutes:
                raise SystemExit("resume minutes cannot be reduced")
            args = dataclasses.replace(args, minutes=minutes)
        expected = pathlib.Path(args.out)
        requested = pathlib.Path(name if name and name.startswith("runs/")
                                 else f"runs/{name}") if name else expected
        if requested != expected:
            raise SystemExit(f"resume belongs to {expected}, not {requested}")
        args.resume = resume
    else:
        if not name:
            raise SystemExit("pass out=<name>")
        args = dataclasses.replace(config.BASELINE, **over)
        args.git = config.git_sha()
        args.out = name if name.startswith("runs/") else f"runs/{name}"
    if resume:
        if not os.path.isdir(args.out):
            raise SystemExit(f"resume output {args.out} does not exist")
    else:
        if os.path.exists(args.out):
            raise SystemExit(f"{args.out} exists")
        os.makedirs(args.out)
    logf = open(f"{args.out}/train.log", "a" if resume else "w")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            logf.write(s)
            return len(s)
        def flush(self):
            sys.__stdout__.flush()
            logf.flush()
    sys.stdout = sys.stderr = Tee()
    if resume:
        print(f"[resume] {resume}", flush=True)
    else:
        print(f"[train] {args.out} at {args.git} seed={args.seed} "
              f"{over or 'baseline'}", flush=True)
    if args.note:
        print(f"[train] {args.note}", flush=True)
    if args.gen_workers == 0:
        cores = physical_cpus()
        args.gen_workers = len(cores) or (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    diag_rng = np.random.default_rng(args.seed ^ 0xD1A6_0571)
    dev = torch.device(args.device)
    if dev.type != "cuda":
        raise SystemExit(f"device must be a CUDA device, got {args.device!r}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; training requires a working GPU")
    if args.replay_ratio <= 0.0:
        raise SystemExit("replay_ratio must be positive")
    if args.target_every <= 0.0:
        raise SystemExit("target_every must be positive minutes")
    if args.gen_solves <= 0 or args.gen_workers <= 0:
        raise SystemExit("gen_solves and resolved gen_workers must be positive")
    torch.cuda.set_device(dev)
    if args.train_stream_priority > 0:
        raise SystemExit("train_stream_priority must be zero or negative")
    if args.train_stream_priority < 0:
        default_stream = torch.cuda.current_stream(dev)
        train_stream = torch.cuda.Stream(
            device=dev, priority=args.train_stream_priority)
        train_stream.wait_stream(default_stream)
        torch.cuda.set_stream(train_stream)
        print(f"[train] CUDA stream priority {args.train_stream_priority}", flush=True)

    value = Net().to(dev)
    if args.init_weights:
        value.load_state_dict(load_checkpoint(args.init_weights).state_dict())
    opt = torch.optim.Adam(value.parameters(), lr=args.lr)
    value.push()
    target_state = cpu_state(value)
    buf = Buffer(args.cap, args.cap * args.cfgs_per_row)
    buf.x.fill(0)
    import gpu_batch
    gpu_batch.warmup(dev)
    batcher = gpu_batch.make_batch
    # One training step and one probe at a training-sized batch, so the
    # caching allocator holds that peak before the farm carves. Configs in a
    # batch are `rows × cfgs_per_row`. The solve slot's config cap is not a
    # batch width: at s=512 it is thousands, and a dummy that wide does not
    # fit next to the intern table.
    # 2048 is the intern-table floor: a 2048× table does not fit the encoder.
    torch.cuda.reset_peak_memory_stats(dev)
    names = tuple(warchest.ENT_NAMES)
    n = max(args.batch, 2048)
    k = n * args.cfgs_per_row
    x = torch.zeros(2 * n, PUBFEAT, device=dev)
    phi = torch.zeros(k, CFEAT, device=dev)
    w = torch.ones(k, device=dev)
    seg = torch.arange(k, device=dev) % (2 * n)
    y = torch.zeros(k, device=dev)
    parts = (x, phi, w, seg, y, 2 * n, None)
    opt.zero_grad(set_to_none=True)
    losses(value, *parts, wp=0.0).backward()
    opt.step()
    forward_values(value, parts)
    torch.cuda.synchronize(dev)
    if checkpoint:
        value.load_state_dict(checkpoint["value"])
        opt.load_state_dict(checkpoint["optimizer"])
        target_state = checkpoint["target"]
        publish_state(target_state)
        rng.bit_generator.state = checkpoint["numpy_rng"]
        diag_rng.bit_generator.state = checkpoint["diag_numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        buf.load_state_dict(checkpoint["buffer"])
    peak = torch.cuda.max_memory_reserved(dev)
    print(f"[train] torch peak {peak / (1 << 20):.0f} MiB reserved on {dev} "
          f"(rows={n} configs={k}); farm carves mem_get_info free",
          flush=True)
    print(f"[train] search inference on cuda:{args.gen_devices}, "
          f"training on {dev}", flush=True)

    total = args.minutes * 60.0
    if args.snapshot_every <= 0:
        raise SystemExit("snapshot_every must be positive minutes")
    snap_gap = args.snapshot_every * 60.0
    elapsed = float(checkpoint["elapsed"]) if checkpoint else 0.0
    next_snap = float(checkpoint["next_snapshot"]) if checkpoint else snap_gap
    t0 = time.time() - elapsed
    epoch = int(checkpoint["epoch"]) if checkpoint else 0
    if checkpoint:
        epoch = max(epoch, last_epoch(args))
    progress = checkpoint["progress"] if checkpoint else {
        "sog_start": None,
        "sog_solves": 0,
        "optimizer_rows": 0,
        "generated_rows": 0,
        "next_target": None,
        "farm_runs": 0,
        "totals": {},
    }
    sog_t0 = (t0 + progress["sog_start"]
              if progress["sog_start"] is not None else None)
    sog_solves = int(progress["sog_solves"])
    probe = None

    # Snapshots are also checkpoints. Evaluate packed bots separately after the
    # run, outside the training clock.
    snaps = checkpoint["snapshots"] if checkpoint else []

    def snapshot(label, el):
        # "init" and "final" are the two the reader always wants named; the rest
        # are numbered, and the manifest carries the time each was taken at.
        path = f"{args.out}/snap_{len(snaps):04d}.pt"
        entry = {"label": label, "t": round(el, 1),
                 "file": os.path.basename(path)}
        snaps.append(entry)
        cfg = dataclasses.asdict(args)
        cfg["resume"] = ""
        state = {
            "value": value.state_dict(),
            "optimizer": opt.state_dict(),
            "target": target_state,
            "numpy_rng": rng.bit_generator.state,
            "diag_numpy_rng": diag_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "buffer": buf.state_dict(),
            "elapsed": float(el),
            "next_snapshot": float(el + snap_gap),
            "epoch": epoch,
            "snapshots": snaps,
            "progress": progress,
            "cfg": cfg,
            "t": round(el, 1),
            "label": label,
            "git": args.git,
            "search": {"s": args.s, "c": args.c, "cfr": args.cfr},
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        write_log(args, snaps)
        print(f"[t={el:6.1f}s] --- snapshot {snaps[-1]['file']} ({label}) ---", flush=True)

    def tick(rec=None, line=None):
        """Log one epoch if given, then snapshot if due. Both generators use this."""
        nonlocal epoch, next_snap
        if rec is not None:
            rec.setdefault("t", round(time.time() - t0, 1))
            rec.setdefault("epoch", epoch)
            rec.setdefault("buf", len(buf))
            rec.setdefault("lr", opt.param_groups[0]["lr"])
            append_epoch(args, rec)
            print(line, flush=True)
            epoch += 1
        now = time.time()
        if now - t0 >= next_snap:
            snapshot(f"s{len(snaps)}", now - t0)
            next_snap = now - t0 + snap_gap

    def fit(nsteps, deadline=None):
        """Adam updates on the shared replay. Returns (loss, wall_s, stats)."""
        if sog_t0 is not None:
            opt.param_groups[0]["lr"] = scheduled_lr(
                args.lr, args.lr_final, time.time() - sog_t0,
                total - warm, args.lr_stable_frac)
        if nsteps < 1 or len(buf) < args.batch:
            return float("nan"), 0.0, {}
        tt = time.time()
        lv, st = train_steps(
            value, opt, buf, nsteps, args.batch, rng, dev,
            recent_mix=args.recent_mix, recent_frac=args.recent_frac,
            profile_cuda=os.environ.get("WARCHEST_TRAIN_PROFILE") == "1",
            batch_fn=batcher, policy_w=args.policy_w, deadline=deadline)
        return lv, time.time() - tt, st

    def run_search_pipeline():
        """Overlap small GT-CFR batches with each other and with training."""
        nonlocal probe, sog_solves, target_state

        deadline = t0 + total
        if time.time() >= deadline:
            return
        # One process, one driver a card and a worker per core. A process per
        # core could not batch inference at all: the solves have to share an
        # address space for their leaf rows to end up in one batch.
        farm = warchest.SolveFarm(
            args.seed + 1_000_003 * progress["farm_runs"],
            args.gen_workers,
            s=args.s,
            c=args.c,
            batch=args.round_batch,
            rounds=args.rounds,
            explore=args.explore,
            random_draft=args.random_draft,
            cfr=args.cfr,
            p_td1=args.p_td1,
            query_rate=args.query_rate,
            recursive_rate=args.recursive_rate,
            devices=[int(d) for d in args.gen_devices.split(",")])

        progress["farm_runs"] += 1
        optimizer_rows = int(progress["optimizer_rows"])
        generated_rows = int(progress["generated_rows"])
        window = collections.Counter()
        totals = collections.Counter(progress["totals"])
        window_shapes = []
        window_targets = []
        window_target_weights = []
        window_sample_ages = []
        window_sample_delays = []
        # The farm's round counters are cumulative, so an epoch's figures are
        # the difference against the last report and not an average since the
        # farm started.
        round_at = dict.fromkeys(ROUND_KEYS, 0)
        stage_names = warchest.stage_names()
        ent_at = [0] * 8
        next_report = time.time() + 10.0
        next_target = t0 + float(progress["next_target"])

        def save_progress():
            progress.update({
                "sog_solves": sog_solves,
                "optimizer_rows": optimizer_rows,
                "generated_rows": generated_rows,
                "next_target": next_target - t0,
                "totals": dict(totals),
            })

        while True:
            now = time.time()
            if now >= deadline:
                break
            gen_t = time.time()
            data = farm.collect(args.gen_solves)
            gen_s = time.time() - gen_t

            ta = time.time()
            n = ingest(buf, data)
            add_s = time.time() - ta
            cy = np.clip(np.asarray(data["cy"], np.float32), -1.0, 1.0)
            cw = np.asarray(data["cw"], np.float32)
            window_targets.append(cy)
            window_target_weights.append(cw)

            solves = int(data["solves"])
            sog_solves += solves
            generated_rows += n
            window["results"] += 1
            window["rows"] += n
            window["solves"] += solves
            window["target_n"] += cy.size
            window["target_sum"] += float(cy.sum(dtype=np.float64))
            window["target_square_sum"] += float(
                np.square(cy.astype(np.float64)).sum())
            window["gen_s"] += gen_s
            window["add_s"] += add_s
            window_shapes.extend(data.get("shapes") or [])
            for name in (
                    "games", "decisions", "horizon_hits",
                    "white_wins", "black_wins", "draws",
                    "plays_attack", "plays_pass", "plays_deploy",
                    "plays_bolster", "plays_maneuver", "plays_recruit",
                    "plays_claim_initiative", "configs", "query_rows", "dropped"):
                amount = int(data.get(name, 0))
                totals[name] += amount
                window[name] += amount
            debt = max(0.0, args.replay_ratio * generated_rows - optimizer_rows)
            nsteps = int(debt // args.batch) if len(buf) >= args.batch else 0
            lv, train_s, train_stat = fit(nsteps, deadline)
            trained = train_stat.get("steps", 0)
            if trained:
                optimizer_rows += trained * args.batch
                window["loss_sum"] += lv * trained
                window["train_steps"] += trained
                window["policy_steps"] += train_stat["policy_steps"]
                for key in (
                        "policy_loss", "policy_target_entropy", "policy_prior_entropy",
                        "policy_search_kl"):
                    window[f"{key}_sum"] += train_stat[f"{key}_sum"]
                window["batch_configs"] += train_stat["batch_configs"]
                window["gpu_forward_s"] += train_stat["gpu_forward_s"]
                window["gpu_backward_s"] += train_stat["gpu_backward_s"]
                window["grad_clipped"] += train_stat["grad_clipped"]
                window["grad_norm_sum"] += train_stat["grad_norm_sum"]
                window["grad_norm_max"] = max(
                    window["grad_norm_max"], train_stat["grad_norm_max"])
                for key in ("sample_warm", "sample_play", "sample_query",
                            "sample_warm_delay_sum", "sample_play_delay_sum",
                            "sample_query_delay_sum", "sample_td1_targets",
                            "sample_targets"):
                    window[key] += train_stat[key]
                window_sample_ages.extend(train_stat["sample_ages"])
                window_sample_delays.extend(train_stat["sample_delays"])
                window["zero_sum_max"] = max(
                    window["zero_sum_max"], train_stat["zero_sum_max"])
                window["zero_sum_square_sum"] += train_stat["zero_sum_square_sum"]
                window["zero_sum_n"] += train_stat["zero_sum_n"]
            window["train_s"] += train_s

            now = time.time()
            if now >= deadline:
                save_progress()
                break
            if now >= next_target:
                # `push` bumps the version the farm watches; its threads
                # pick the new weights up at their next chunk.
                value.push()
                target_state = cpu_state(value)
                if len(buf) >= 2048:
                    probe = batcher(buf.sample(2048, diag_rng), diag_rng, dev)
                print(
                    f"[t={now - t0:6.1f}s] --- target network refresh ---",
                    flush=True)
                while next_target <= now:
                    next_target += args.target_every * 60.0
            sog_elapsed = max(0.0, now - sog_t0)
            save_progress()
            if now < next_report:
                tick()
                continue
            next_report = now + 10.0
            steps = int(window["train_steps"])
            lv = window["loss_sum"] / max(steps, 1)
            if probe is None and len(buf) >= 2048:
                probe = batcher(buf.sample(2048, diag_rng), diag_rng, dev)
            diag = diagnostics(value, buf, probe, args.batch, diag_rng, dev, batcher,
                               args.recent_frac)
            target_n = max(int(window["target_n"]), 1)
            target_mean = window["target_sum"] / target_n
            target_var = max(
                0.0,
                window["target_square_sum"] / target_n
                - target_mean * target_mean)
            targets = np.concatenate(window_targets)
            target_weights = np.concatenate(window_target_weights)
            weight_mass = max(float(target_weights.sum()), 1e-12)
            belief_mean = float(np.dot(targets, target_weights) / weight_mass)
            belief_var = max(float(np.dot((targets - belief_mean) ** 2,
                                          target_weights) / weight_mass), 0.0)
            target_q = np.quantile(targets, [0.05, 0.5, 0.95])
            sample_ages = (np.concatenate(window_sample_ages)
                           if window_sample_ages else np.zeros(1))
            sample_delays = (np.concatenate(window_sample_delays)
                             if window_sample_delays else np.zeros(1))
            replay = buf.replay_stats()
            sample_n = max(window["sample_warm"] + window["sample_play"]
                           + window["sample_query"], 1)
            policy_steps = max(int(window["policy_steps"]), 1)
            policy = {
                key: window[f"{key}_sum"] / policy_steps
                for key in (
                    "policy_loss", "policy_target_entropy", "policy_prior_entropy",
                    "policy_search_kl")
            }
            weight_norm = float(torch.sqrt(sum(
                p.detach().float().square().sum() for p in value.parameters())))
            dec = max(int(window["decisions"]), 1)
            games = max(int(window["games"]), 1)
            raw_sps = sog_solves / max(sog_elapsed, 1e-9)
            gen_s = window["gen_s"] / max(window["results"], 1)
            train_s = window["train_s"]
            now_at = {k: int(data[k]) for k in ROUND_KEYS}
            rounds = max(now_at["rounds"] - round_at["rounds"], 1)
            per_round = {k: (now_at[k] - round_at[k]) / rounds
                         for k in ROUND_KEYS[1:]}
            hits = now_at["budget_hits"] - round_at["budget_hits"]
            round_at = now_at
            leaf = warchest.leaf_breakdown()
            leaf_breakdown = {
                "ms_per_round": {
                    name: round(value / rounds, 3)
                    for name, value in zip(stage_names[:-3], leaf[:-3])
                },
                "bytes_per_round": {
                    name: round(value * 1e6 / rounds)
                    for name, value in zip(stage_names[-3:-1], leaf[-3:-1])
                },
            }
            stage_line = "/".join(
                f"{name}={value:g}"
                for name, value in leaf_breakdown["ms_per_round"].items()
                if value > 0)
            names = tuple(warchest.ENT_NAMES)
            now_ent = [int(x) for x in (data.get("entity_hits") or [0] * 8)]
            ent_hits = [now_ent[i] - ent_at[i] for i in range(8)]
            ent_at = now_ent
            bounds = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536)
            if window_shapes:
                a = np.asarray(window_shapes, np.uint32)
                def pct(col, q):
                    v = np.sort(a[:, col])
                    return int(v[int(round((len(v) - 1) * q))])
                shape = {
                    names[i]: {
                        "p50": pct(i, 0.50),
                        "p90": pct(i, 0.90),
                        "p99": pct(i, 0.99),
                        "max": int(a[:, i].max()),
                    }
                    for i in range(8)
                }
                node_histogram = {}
                stop_census = {}
                for kind_id, kind in enumerate(warchest.SOLVE_KIND_NAMES):
                    ka = a[a[:, 9] == kind_id]
                    hist = {}
                    for lo, hi in zip(bounds[:-1], bounds[1:]):
                        hist[f"{lo}-{hi - 1}"] = int(
                            ((ka[:, 0] >= lo) & (ka[:, 0] < hi)).sum())
                    hist[f"{bounds[-1]}+"] = int((ka[:, 0] >= bounds[-1]).sum())
                    node_histogram[kind] = hist
                    stops = {}
                    for stop_id, stop in enumerate(warchest.STOP_NAMES):
                        nodes = ka[ka[:, 8] == stop_id, 0]
                        if nodes.size:
                            nodes.sort()
                            stops[stop] = {
                                "count": int(nodes.size),
                                "node_p50": int(nodes[int(round(
                                    (nodes.size - 1) * 0.5))]),
                            }
                    stop_census[kind] = stops
            else:
                shape = {n: {"p50": 0, "p90": 0, "p99": 0, "max": 0} for n in names}
                empty_hist = {
                    **{f"{lo}-{hi - 1}": 0
                       for lo, hi in zip(bounds[:-1], bounds[1:])},
                    f"{bounds[-1]}+": 0,
                }
                node_histogram = {
                    k: dict(empty_hist) for k in warchest.SOLVE_KIND_NAMES}
                stop_census = {k: {} for k in warchest.SOLVE_KIND_NAMES}
            rec = {
                "t": round(now - t0, 1),
                "epoch": epoch,
                "phase": "sog",
                "games": int(window["games"]),
                # Decisive against drawn, per epoch. A finished game is the
                # earliest evidence that self-play is going anywhere at all, and
                # a run that only ever draws is failing differently from one
                # that never finishes a game.
                "white_wins": int(window["white_wins"]),
                "black_wins": int(window["black_wins"]),
                "draws": int(window["draws"]),
                "decisions": int(window["decisions"]),
                "rows": int(window["rows"]),
                "solves": int(window["solves"]),
                "loss": round(lv, 5),
                "total_loss": round(lv + args.policy_w * policy["policy_loss"], 5),
                "loss_old": round(diag["loss_old"], 5),
                "loss_new": round(diag["loss_new"], 5),
                "zero_sum_max": round(window["zero_sum_max"], 5),
                "zero_sum_rms": round((window["zero_sum_square_sum"]
                                        / max(window["zero_sum_n"], 1)) ** 0.5, 5),
                "grad_norm": round(window["grad_norm_sum"] / max(steps, 1), 4),
                "grad_norm_max": round(window["grad_norm_max"], 4),
                "weight_norm": round(weight_norm, 4),
                "grad_clip_frac": round(
                    window["grad_clipped"] / max(steps, 1), 4),
                "horizon_frac": round(window["horizon_hits"] / games, 3),
                # How many solves shared a forward pass. It should sit near the
                # thread count; well below means the round is waiting on
                # stragglers instead of batching them. A round carries
                # `round_batch` regret updates, so a solve rides in ceil(64 /
                # round_batch) rounds rather than sixty-five, and the two
                # figures below rise with the knob: none of the three compares
                # across a change to it.
                "calls_per_round": round(per_round["round_calls"], 2),
                "rows_per_round": round(per_round["round_rows"], 1),
                # Milliseconds a round spends inside the device backend — the
                # batch plus the concatenation and split around it. The rest of
                # a round is CFR on the cores.
                "device_ms_per_round": round(
                    1e-6 * per_round["round_nanos"], 2),
                "leaf_breakdown": leaf_breakdown,
                # Rows the query solver produced, i.e. targets taken off
                # the line of play. Zero means the coverage path is dead.
                "query_rows": int(window["query_rows"]),
                "dropped": int(window["dropped"]),
                "plays": {
                    name: int(window[f"plays_{name}"])
                    for name in (
                        "attack", "pass", "deploy", "bolster",
                        "maneuver", "recruit", "claim_initiative")
                },
                "configs": round(window["configs"] / dec, 1),
                "steps": steps,
                "optimizer_steps": optimizer_rows // args.batch,
                "optimizer_rows": optimizer_rows,
                "optimizer_debt": round(
                    max(0.0, args.replay_ratio * generated_rows - optimizer_rows), 1),
                "replay_rows": generated_rows,
                "rows_per_s": round(
                    generated_rows / max(sog_elapsed, 1e-9), 1),
                "effective_train_ratio": round(
                    optimizer_rows / max(sog_solves, 1), 3),
                "train_row_ratio": round(
                    optimizer_rows / max(generated_rows, 1), 3),
                "tgt_mean": round(target_mean, 4),
                "tgt_std": round(target_var ** 0.5, 4),
                "tgt_belief_mean": round(belief_mean, 4),
                "tgt_belief_std": round(belief_var ** 0.5, 4),
                "tgt_p05": round(float(target_q[0]), 4),
                "tgt_p50": round(float(target_q[1]), 4),
                "tgt_p95": round(float(target_q[2]), 4),
                "tgt_abs95_frac": round(float(np.mean(np.abs(targets) >= 0.95)), 4),
                "probe_std": round(diag["probe_std"], 4),
                "value_outcome_rmse": round(diag["value_outcome_rmse"], 4),
                "value_outcome_mae": round(diag["value_outcome_mae"], 4),
                "value_outcome_bias": round(diag["value_outcome_bias"], 4),
                "value_outcome_corr": round(diag["value_outcome_corr"], 4),
                "value_calibration_slope": round(diag["value_calibration_slope"], 4),
                "gen_s": round(gen_s, 2),
                "train_s": round(train_s, 2),
                "add_s": round(window["add_s"], 2),
                "gpu_forward_s": round(window["gpu_forward_s"], 2),
                "gpu_backward_s": round(window["gpu_backward_s"], 2),
                "batch_configs": round(
                    window["batch_configs"] / max(steps, 1), 1),
                "buf_s": round(buf.span_seconds(), 1),
                "sample_age_mean": round(float(sample_ages.mean()), 1),
                "sample_age_p50": round(float(np.quantile(sample_ages, 0.5)), 1),
                "sample_age_p90": round(float(np.quantile(sample_ages, 0.9)), 1),
                "sample_delay_mean": round(float(sample_delays.mean()), 1),
                "sample_delay_p90": round(float(np.quantile(sample_delays, 0.9)), 1),
                "sample_warm_delay": round(
                    window["sample_warm_delay_sum"] / max(window["sample_warm"], 1), 1),
                "sample_play_delay": round(
                    window["sample_play_delay_sum"] / max(window["sample_play"], 1), 1),
                "sample_query_delay": round(
                    window["sample_query_delay_sum"] / max(window["sample_query"], 1), 1),
                "sample_warm_frac": round(window["sample_warm"] / sample_n, 4),
                "sample_play_frac": round(window["sample_play"] / sample_n, 4),
                "sample_query_frac": round(window["sample_query"] / sample_n, 4),
                "sample_td1_target_frac": round(
                    window["sample_td1_targets"] / max(window["sample_targets"], 1), 5),
                **{key: round(value, 5) for key, value in replay.items()},
                "solves_per_s": round(raw_sps, 1),
                **{key: round(value, 5) for key, value in policy.items()},
                "policy_weighted_loss": round(args.policy_w * policy["policy_loss"], 5),
                "budget_hits": int(hits),
                "entity_hits": {names[i]: ent_hits[i] for i in range(8)},
                "slots": int(data.get("slots", 0)),
                "slots_used": int(data.get("slots_used", 0)),
                "slots_per_card": int(data.get("slots_per_card", 0)),
                "slot_bytes": int(data.get("slot_bytes", 0)),
                "shape": shape,
                "node_histogram": node_histogram,
                "stop_census": stop_census,
            }
            rec["budget_hit_rate"] = round(
                rec["budget_hits"] / max(len(window_shapes), rec["solves"], 1), 3)
            tick(rec,
                f"[t={rec['t']:6.1f}s] GT-CFR solves={sog_solves} "
                f"rate={raw_sps:.1f}/s rows={rec['rows']} "
                f"games={rec['games']} "
                f"W{rec['white_wins']}/B{rec['black_wins']}/D{rec['draws']} "
                f"qrows={rec['query_rows']} "
                f"L={lv:.5f} L/var={lv / max(target_var, 1e-9):.2f} "
                f"Lp={rec['policy_loss']:.3f} "
                f"tgt={target_mean:+.3f}/{target_var ** 0.5:.3f} "
                f"gen={gen_s:.2f}s train={train_s:.2f}s "
                f"gpu={window['gpu_forward_s'] + window['gpu_backward_s']:.2f}s "
                f"slots={rec['slots_used']}/{rec['slots']} "
                f"spc={rec['slots_per_card']} "
                f"slot={rec['slot_bytes'] / (1 << 20):.1f}MiB "
                f"hits={rec['budget_hits']} "
                f"hit_rate={rec['budget_hit_rate']} "
                f"stages={stage_line} "
                f"ehits={'/'.join(str(ent_hits[i]) for i in range(8))} "
                f"p90={'/'.join(str(shape[n]['p90']) for n in names)}")
            window.clear()
            window_shapes.clear()
            window_targets.clear()
            window_target_weights.clear()
            window_sample_ages.clear()
            window_sample_delays.clear()
        # Dropping the farm stops its threads once they finish the solve they
        # are in, which is also what flushes their last rows.
        del farm
        save_progress()

        elapsed = max(deadline - sog_t0, 1e-9)
        print(
            f"[GT-CFR-summary] solves={sog_solves} "
            f"optimizer_rows={optimizer_rows} "
            f"rate={sog_solves / elapsed:.1f}/s "
            f"horizon={totals['horizon_hits'] / max(totals['games'], 1):.2f} "
            f"games={totals['games']} "
            f"W{totals['white_wins']}/B{totals['black_wins']}/D{totals['draws']}",
            flush=True)

    print(f"[cfg] PUBFEAT={PUBFEAT} CFEAT={CFEAT} architecture=gt-cfr "
          f"s={args.s} c={args.c} "
          f"budget={total:.0f}s warm={args.warm_minutes:g}min "
          f"lr={args.lr:g}->{args.lr_final:g} stable={args.lr_stable_frac:g} "
          f"snapshot_every={args.snapshot_every:g}min device={dev} "
          f"draft={'random' if args.random_draft else 'starter'} "
          f"replay_ratio={args.replay_ratio} "
          f"recent_mix={args.recent_mix}/{args.recent_frac} "
          f"canonical_views=2 cap={args.cap} "
          f"matmul={torch.get_float32_matmul_precision()}", flush=True)

    if not checkpoint:
        snapshot("init", 0.0)
    warm = args.warm_minutes * 60.0
    if not 0.0 <= warm <= total:
        raise SystemExit("warm_minutes must be between zero and the run length")
    if args.lr <= 0.0 or args.lr_final <= 0.0 or args.lr_final > args.lr:
        raise SystemExit("lr_final must be positive and no greater than lr")
    if not 0.0 <= args.lr_stable_frac <= 1.0:
        raise SystemExit("lr_stable_frac must be between zero and one")
    if sog_t0 is None:
        while True:
            el = time.time() - t0
            if el >= warm:
                break
            tg = time.time()
            d = warchest.gen_data(
                args.warm_games, args.seed * 1_000_003 + epoch,
                explore=args.explore, random_draft=args.random_draft,
                temp=args.temp)
            gen_s = time.time() - tg
            n = ingest(buf, d, warm=True)
            steps = max(1, n // args.batch) if len(buf) >= args.batch else 0
            lv, train_s, _ = fit(steps)
            cy = np.clip(np.asarray(d["cy"], np.float32), -1.0, 1.0)
            rec = {
                "t": round(time.time() - t0, 1), "epoch": epoch, "phase": "warm",
                "games": int(d.get("games", 0)), "rows": n,
                "loss": round(float(lv), 5) if steps else None,
                "tgt_mean": round(float(cy.mean()) if cy.size else 0.0, 4),
                "tgt_std": round(float(cy.std()) if cy.size else 0.0, 4),
                "horizon_frac": round(int(d.get("horizon_hits", 0)) /
                                      max(int(d.get("games", 0)), 1), 3),
                "gen_s": round(gen_s, 2), "train_s": round(train_s, 2),
            }
            tick(rec,
                f"[t={rec['t']:6.1f}s] warm ep{epoch:3d} games={rec['games']:4d} "
                f"rows={n:6d} L={lv if steps else float('nan'):.5f} "
                f"tgt={rec['tgt_mean']:+.3f}/{rec['tgt_std']:.3f} "
                f"gen={gen_s:.1f}s train={train_s:.1f}s")
        value.push()
        target_state = cpu_state(value)
        sog_t0 = time.time()
        progress["sog_start"] = sog_t0 - t0
        progress["next_target"] = (sog_t0 - t0) + args.target_every * 60.0
    # Keep warm rows in replay; the FIFO ages them out.
    run_search_pipeline()

    snapshot("final", time.time() - t0)
    subprocess.run([sys.executable, str(ROOT / "tools" / "arena.py"),
                    "pack", args.out, "--out", str(ROOT / "bots")], check=True)


if __name__ == "__main__":
    main()
