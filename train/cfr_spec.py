#!/usr/bin/env python
"""The CFR step as executable specification (work package B, step 1).

Reads a serialized solve job (the byte format of engine/src/serialize.rs) and
its weights, and replays the whole solve with torch ops: the exact formulas
and the exact phase order of engine/src/search.rs. The CUDA phase kernels are
written against this spec; `--check` compares the spec's outputs with the CPU
solver's reference dump (engine/examples/oracle_dump.rs) within tolerance.

Not shipped: this is the oracle harness for the GPU service, not part of the
generation path.

Usage:
    python train/cfr_spec.py --check <oracle-dir> [solve-index ...]
    python train/cfr_spec.py --job job.bin --weights weights.bin   # run only
"""

import struct

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6       # regret-matching floor (search.rs)
LN_EPS = 1e-5    # LayerNorm epsilon (net.rs, matches torch default)
SMOOTH = 1e-30   # belief-normalization floor (rebel.rs)

# Frozen encoding geometry (rebel.rs / units.rs). The spec reads the job
# arrays, not the Rust constants, but the feature offsets live here because
# they are layout, not data.
N_HEXES, NTYPE, NSLOT = 37, 10, 5
HEX_FACTS, HEX_CH = 6, 6 + 10
PILE_COUNTS, CARD_FEATS = 4, 25
OFF_PILES = N_HEXES * HEX_CH              # 592
OFF_CARDS = OFF_PILES + NTYPE * PILE_COUNTS  # 632
OFF_LOOSE = OFF_CARDS + NTYPE * CARD_FEATS   # 882
LOOSE = 15                                 # 2*6 + 3
PUBFEAT = OFF_LOOSE + LOOSE                # 897
CCOUNTS = 3 * NSLOT                       # 15
CFEAT = CCOUNTS + 1                       # 16
AFEAT = 39 + 3 * (N_HEXES + 1) + 2 * (NTYPE + 1) + 1  # 176
AOFF_PAYS = 39 + 3 * (N_HEXES + 1)         # 153


# ---------------------------------------------------------------- job reader


class Job:
    """The serialized job, as numpy arrays. Field order = serialize.rs."""

    def __init__(self, path):
        self.b = open(path, "rb").read()
        self.at = 0
        magic, ver = self.u32(), self.u32()
        assert magic == 0x57434A33 and ver == 3, f"bad job magic/version {magic:x}/{ver}"
        self.depth, self.iters = self.u32(), self.u32()
        self.snapshots = bool(self.take(1)[0])
        self.alpha, self.beta, self.gamma, self.predict, self.warm = (
            self.f32(), self.f32(), self.f32(), self.f32(), self.f32())
        self.snap_iters = self.u32s()
        self.nodes, self.ncfg, self.rows, self.pubfeat, self.ncells = (
            self.u32(), self.u32(), self.u32(), self.u32(), self.u32())
        self.node_kind = self.u8s()
        self.node_player = self.u8s()
        self.node_leaf = self.u8s()
        self.node_child_start = self.u32s()
        self.node_child = self.u32s()
        self.obs_off = self.u32s()
        self.obs_start = self.u32s()
        self.obs_act = self.u32s()
        self.obs_child = self.u32s()
        self.legal_bits = self.u8s()
        self.trans = self.i32s()
        self.draw_off = self.u32s()
        self.draw_to = self.u32s()
        self.draw_p = self.f32s()
        self.draw_row_off = self.u32s()
        self.draw_row_start = self.u32s()
        self.cfg_off = self.u32s()
        self.reach_off = self.u32s()
        self.soff = self.u32s()
        # Reverse (gather) transitions; this spec keeps the forward loops,
        # so they are read to stay in sync with the byte order only.
        self.node_parent = self.u32s()
        self.rev_row_of = self.u32s()
        self.rev_start = self.u32s()
        self.rev_src = self.u32s()
        self.rev_cell = self.u32s()
        self.rvd_row_of = self.u32s()
        self.rvd_start = self.u32s()
        self.rvd_src = self.u32s()
        self.rvd_p = self.f32s()
        self.leaf_rows = self.u32s()
        self.inner_rows = self.u32s()
        self.term_leaves = self.u32s()
        self.terminal_utility = self.f32s()
        self.leaf_coff = self.u32s()
        self.leaf_cidx = self.u32s()
        self.leaf_xpub = self.f32s()
        self.cphi = self.f32s()
        self.bfs_order = self.u32s()
        self.level_start = self.u32s()
        self.ids = self.u8s()
        self.root0 = self.f32s()
        self.root1 = self.f32s()
        self.carried = []
        for _ in range(self.u32()):
            self.carried.append([self.f32s(), self.f32s()])
        assert self.at == len(self.b), "trailing bytes in job"
        # Derived geometry.
        n = self.nodes
        self.nc = np.zeros((n, 2), np.int64)
        for i in range(n):
            for p in range(2):
                self.nc[i, p] = self.cfg_off[2 * i + p + 1] - self.cfg_off[2 * i + p]
        voff = [0]
        for i in range(n):
            voff.append(voff[-1] + max(self.nc[i, 0], self.nc[i, 1]))
        self.voff = np.array(voff, np.int64)
        self.nleaf = len(self.leaf_rows)
        self.n_inner = self.rows - self.nleaf
        self.legal = np.zeros(self.ncells, bool)
        for bi in range(len(self.legal_bits)):
            byte = int(self.legal_bits[bi])
            for bit in range(8):
                cell = bi * 8 + bit
                if cell < self.ncells and (byte >> bit) & 1:
                    self.legal[cell] = True
        # Per-node action counts from the obs segments: a segment's last
        # boundary is the node's action count (segments are node-relative).
        self.na = np.zeros(n, np.int64)
        for i in range(n):
            a0, a1 = self.obs_off[i], self.obs_off[i + 1]
            if a1 > a0:
                self.na[i] = self.obs_start[a1 - 1]
        self.action_off = np.concatenate([[0], np.cumsum(self.na)])
        self.child = [self.node_child[self.node_child_start[i]:self.node_child_start[i + 1]]
                      for i in range(n)]

    def take(self, n):
        s = self.b[self.at:self.at + n]
        assert len(s) == n, "truncated job"
        self.at += n
        return s

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def f32(self):
        return struct.unpack("<f", self.take(4))[0]

    def u8s(self):
        return np.frombuffer(self.take(self.u32()), np.uint8)

    def u32s(self):
        n = self.u32()
        return np.frombuffer(self.take(4 * n), np.uint32).astype(np.int64)

    def u64s(self):
        n = self.u32()
        return np.frombuffer(self.take(8 * n), np.uint64)

    def i32s(self):
        n = self.u32()
        return np.frombuffer(self.take(4 * n), np.int32).astype(np.int64)

    def i8s(self):
        n = self.u32()
        return np.frombuffer(self.take(n), np.int8).astype(np.int64)

    def f32s(self):
        n = self.u32()
        return np.frombuffer(self.take(4 * n), np.float32)


# ---------------------------------------------------------------- the spec


def load_weights(path):
    """The export_weights.py flat format, sliced into the named matrices.
    Every matrix is row-major [in, out]; `w` is the concatenation of
    (wd0, wd1, wid, wpile, w0, w1, wb, wc, wh1, wh2, wg, wu, wq, wk, wp)."""
    raw = open(path, "rb").read()
    at = [0]

    def u32():
        v = struct.unpack("<I", raw[at[0]:at[0] + 4])[0]
        at[0] += 4
        return v

    def f32s():
        n = u32()
        v = np.frombuffer(raw[at[0]:at[0] + 4 * n], np.float32)
        at[0] += 4 * n
        return v

    nd = u32()
    dims = [u32() for _ in range(nd)]
    w = f32s()
    b = f32s()
    ln = f32s()
    if dims[0] == 3:
        return _slice_v3(dims, w, b, ln)
    (pub, h, hd, cf, dg, rk, af0, de, dc, enc) = dims
    af, hf, xd = af0 + de, 4 + de, N_HEXES * (HEX_FACTS + de) + 2 * de + LOOSE
    at[0] = 0

    def take(n):
        v = w[at[0]:at[0] + n]
        at[0] += n
        return v

    W = {}
    W["wd0"] = take(CARD_FEATS * dc).reshape(CARD_FEATS, dc)
    W["wd1"] = take(dc * de).reshape(dc, de)
    W["wid"] = take(20 * de).reshape(20, de)
    W["wpile"] = take((PILE_COUNTS + de) * de).reshape(PILE_COUNTS + de, de)
    W["w0"] = take(xd * h).reshape(xd, h)
    W["w1"] = take(h * hd).reshape(h, hd)
    W["wb"] = take(2 * dg * hd).reshape(2 * dg, hd)
    W["wc"] = take(hf * dg).reshape(hf, dg)
    W["wh1"] = take(dg * dg).reshape(dg, dg)
    W["wh2"] = take(dg * dg).reshape(dg, dg)
    W["wg"] = take(dg * (rk + 1)).reshape(dg, rk + 1)
    W["wu"] = take(hd * rk).reshape(hd, rk)
    W["wq"] = take(af * rk).reshape(af, rk)
    W["wk"] = take(dg * rk).reshape(dg, rk)
    W["wp"] = take(hd * rk).reshape(hd, rk)
    assert at[0] == len(w), f"weight slice mismatch {at[0]}/{len(w)}"
    at[0] = 0

    def takeb(n):
        v = b[at[0]:at[0] + n]
        at[0] += n
        return v

    B = {}
    B["bd0"] = takeb(dc)
    B["bd1"] = takeb(de)
    B["bpile"] = takeb(de)
    B["b0"] = takeb(h)
    B["b1"] = takeb(hd)
    B["bc"] = takeb(dg)
    B["bh1"] = takeb(dg)
    B["bh2"] = takeb(dg)
    B["bg"] = takeb(rk + 1)
    B["bu"] = takeb(rk)
    B["bq"] = takeb(rk)
    B["bk"] = takeb(rk)
    B["bp"] = takeb(rk)
    assert at[0] == len(b), f"bias slice mismatch {at[0]}/{len(b)}"
    L = {
        "ln0w": ln[:h], "ln0b": ln[h:2 * h],
        "ln1w": ln[2 * h:2 * h + hd], "ln1b": ln[2 * h + hd:2 * h + 2 * hd],
    }
    return dims, W, B, L


def _slice_v3(dims, w, b, ln):
    """A v3 (tower) blob, restricted to the classic shape this spec models:
    one card hidden layer, one public layer, no extra head layers, no slot
    hiddens, one residual block. The spec is a debugging oracle; deeper
    towers need its forward loops generalised first."""
    tag, de, dg, rk, hd, nres = dims[:6]
    at = 6
    lists = []
    for _ in range(4):
        n = dims[at]
        lists.append(list(dims[at + 1:at + 1 + n]))
        at += 1 + n
    card, pub, hmlp, slot = lists
    assert at == len(dims), "trailing dims entries"
    assert len(card) == 1 and len(pub) == 1 and not hmlp and not slot and nres == 1, (
        f"cfr_spec models the classic shape only, got card={card} pub={pub} "
        f"hmlp={hmlp} slot={slot} nres={nres}")
    dc, h = card[0], pub[0]
    af, hf, xd = AFEAT + de, 4 + de, N_HEXES * (HEX_FACTS + de) + 2 * de + LOOSE
    at = [0]

    def take(n):
        v = w[at[0]:at[0] + n]
        at[0] += n
        return v

    W = {}
    W["wd0"] = take(CARD_FEATS * dc).reshape(CARD_FEATS, dc)
    W["wd1"] = take(dc * de).reshape(dc, de)
    W["wid"] = take(20 * de).reshape(20, de)
    W["wpile"] = take((PILE_COUNTS + de) * de).reshape(PILE_COUNTS + de, de)
    W["w0"] = take(xd * h).reshape(xd, h)
    W["w1"] = take(h * hd).reshape(h, hd)
    W["wb"] = take(2 * dg * hd).reshape(2 * dg, hd)
    W["wu"] = take(hd * rk).reshape(hd, rk)
    W["wc"] = take(hf * dg).reshape(hf, dg)
    W["wh1"] = take(dg * dg).reshape(dg, dg)
    W["wh2"] = take(dg * dg).reshape(dg, dg)
    W["wg"] = take(dg * (rk + 1)).reshape(dg, rk + 1)
    W["wq"] = take(af * rk).reshape(af, rk)
    W["wk"] = take(dg * rk).reshape(dg, rk)
    W["wp"] = take(hd * rk).reshape(hd, rk)
    assert at[0] == len(w), f"weight slice mismatch {at[0]}/{len(w)}"
    at[0] = 0

    def takeb(n):
        v = b[at[0]:at[0] + n]
        at[0] += n
        return v

    B = {}
    B["bd0"] = takeb(dc)
    B["bd1"] = takeb(de)
    B["bpile"] = takeb(de)
    B["b0"] = takeb(h)
    B["b1"] = takeb(hd)
    B["bu"] = takeb(rk)
    B["bc"] = takeb(dg)
    B["bh1"] = takeb(dg)
    B["bh2"] = takeb(dg)
    B["bg"] = takeb(rk + 1)
    B["bq"] = takeb(rk)
    B["bk"] = takeb(rk)
    B["bp"] = takeb(rk)
    assert at[0] == len(b), f"bias slice mismatch {at[0]}/{len(b)}"
    L = {
        "ln0w": ln[:h], "ln0b": ln[h:2 * h],
        "ln1w": ln[2 * h:2 * h + hd], "ln1b": ln[2 * h + hd:2 * h + 2 * hd],
    }
    return [0, h, hd, CCOUNTS + 1, dg, rk, AFEAT, de, dc, 0], W, B, L


class Solve:
    """One solve, replayed with torch. Fields mirror the Rust solver."""

    def __init__(self, job, dims, W, B, L, dtype=torch.float32):
        self.j = job
        self.dtype = dtype
        (pub, h, hd, cf, dg, rk, af0, de, dc, enc) = dims
        self.h, self.hd, self.dg, self.rk, self.de = h, hd, dg, rk, de
        self.W = {k: torch.tensor(v, dtype=dtype) for k, v in W.items()}
        self.B = {k: torch.tensor(v, dtype=dtype) for k, v in B.items()}
        self.L = {k: torch.tensor(v, dtype=dtype) for k, v in L.items()}
        self.legal = torch.tensor(job.legal)
        self.trans = torch.tensor(job.trans)
        self.obs_act = torch.tensor(job.obs_act)
        self.obs_child = torch.tensor(job.obs_child)
        self.child = [torch.tensor(c, dtype=torch.long) for c in job.child]
        # Arenas, exactly Solver's flat layout.
        self.reach = torch.zeros(int(job.reach_off[-1]), dtype=dtype)
        self.regret = torch.zeros(job.ncells, dtype=dtype)
        self.inst = torch.zeros(job.ncells, dtype=dtype)
        self.cur = torch.zeros(job.ncells, dtype=dtype)
        self.sum_strat = torch.zeros(job.ncells, dtype=dtype)
        self.avg = torch.zeros(job.ncells, dtype=dtype)
        self.vals = torch.zeros(int(job.voff[-1]), dtype=dtype)
        self.snaps = []
        self.steps = [0, 0]
        self.snap_t = 0
        self.last_traverser = None
        # Build GEMMs (once per solve): the card table, the trunk, the config
        # tower, and (warm start) the action towers.
        e = self.cards(job.leaf_xpub[:job.pubfeat], job.ids)
        self.e = e
        self.xb = torch.zeros(job.rows, 2 * dg, dtype=dtype)
        self.h0 = self.trunk(job.leaf_xpub, job.rows, e)
        z, g = self.embed(job.cphi, job.ncfg, e)
        self.z, self.g = z, g
        assert job.warm == 0, "warm start is plan A4; the job no longer carries psi"
        # Strategy init, exactly Solver::new: uniform cur, zero regrets, the
        # reach-weighted uniform seed, snapshot 0 = the uniform average.
        for i in range(job.nodes):
            if job.node_kind[i] != 0:
                continue
            me = int(job.node_player[i])
            nc = int(job.nc[i, me])
            na = int(job.na[i])
            so = int(job.soff[i])
            leg = self.legal[so:so + nc * na].reshape(nc, na)
            u = (leg / leg.sum(1, keepdim=True).clamp(min=1)).reshape(-1)
            self.cur[so:so + nc * na] = u
            self.avg[so:so + nc * na] = u
        self.propagate(self.cur, job.root0, job.root1)
        for i in range(job.nodes):
            if job.node_kind[i] != 0:
                continue
            me = int(job.node_player[i])
            nc = int(job.nc[i, me])
            na = int(job.na[i])
            so = int(job.soff[i])
            r = self.reach_of(i, me)
            self.sum_strat[so:so + nc * na] += (
                r[:, None] * self.cur[so:so + nc * na].reshape(nc, na)).reshape(-1)
        self.snapshot()

    # ------------------------------------------------------- helpers

    def obs_of(self, i):
        j = self.j
        return j.obs_start[j.obs_off[i]:j.obs_off[i + 1]]

    def reach_of(self, i, p):
        j = self.j
        at = j.reach_off[i] + (int(j.nc[i, 0]) if p == 1 else 0)
        return self.reach[at:at + int(j.nc[i, p])]

    def child_of(self, i, a):
        j = self.j
        return int(j.child[i][int(j.obs_child[int(j.action_off[i]) + a])])

    def snapshot(self):
        j = self.j
        if not j.snapshots:
            return
        t = self.snap_t
        self.snap_t += 1
        if t not in set(j.snap_iters):
            return
        self.snaps.append(self.avg.clone())

    # ------------------------------------------------------- the network

    def cards(self, xpub_row, ids):
        c = torch.tensor(np.ascontiguousarray(xpub_row[OFF_CARDS:OFF_LOOSE]),
                         dtype=self.dtype).reshape(NTYPE, CARD_FEATS)
        hid = F.relu(c @ self.W["wd0"] + self.B["bd0"])
        e = hid @ self.W["wd1"] + self.B["bd1"]
        wid = torch.tensor(np.ascontiguousarray(self.W["wid"][ids.astype(np.int64)]), dtype=self.dtype)
        return e + wid

    def trunk(self, xpub, rows, e):
        """assemble + W0 + LN0 + ReLU + W1, exactly net.rs::trunk (h0 is
        pre-norm, pre-bias: b1 and LN1 live in the head)."""
        de = self.de
        x = torch.tensor(np.ascontiguousarray(xpub), dtype=self.dtype).reshape(rows, -1)
        # The stored row interleaves facts and one-hot per hex (HEX_CH wide);
        # the trunk input splits them into two contiguous blocks.
        hx = x[:, :N_HEXES * HEX_CH].reshape(rows, N_HEXES, HEX_CH)
        cnt = x[:, OFF_PILES:OFF_CARDS].reshape(rows, NTYPE, PILE_COUNTS)
        pe = self.B["bpile"] + e @ self.W["wpile"][PILE_COUNTS:]
        ph = cnt @ self.W["wpile"][:PILE_COUNTS]
        pile = F.relu(ph + pe[None]).reshape(rows, 2, NSLOT, de).sum(2).reshape(rows, 2 * de)
        hexf = hx[:, :, :HEX_FACTS]
        hexe = hx[:, :, HEX_FACTS:]
        emb = torch.einsum("rhc,cd->rhd", hexe, e)
        loose = x[:, OFF_LOOSE:OFF_LOOSE + LOOSE]
        xa = torch.cat([hexf.reshape(rows, -1), emb.reshape(rows, -1),
                        pile, loose], 1)
        h0 = F.relu(F.layer_norm(xa @ self.W["w0"] + self.B["b0"], [self.h],
                                 self.L["ln0w"], self.L["ln0b"])) @ self.W["w1"]
        return h0

    def embed(self, phi, n, e):
        de, dg = self.de, self.dg
        phi = torch.tensor(np.ascontiguousarray(phi), dtype=self.dtype).reshape(n, CFEAT)
        seat = phi[:, CCOUNTS].long()
        counts = phi[:, :CCOUNTS].reshape(n, 3, NSLOT).transpose(1, 2)  # [n, 5, 3]
        mine = e[seat.unsqueeze(1) * NSLOT + torch.arange(NSLOT)]
        s = phi[:, CCOUNTS].reshape(-1, 1, 1).expand(-1, NSLOT, 1)
        z = F.relu(torch.cat([counts, s, mine], -1) @ self.W["wc"] + self.B["bc"]).sum(1)
        z = z + F.relu(z @ self.W["wh1"] + self.B["bh1"]) @ self.W["wh2"] + self.B["bh2"]
        g = z @ self.W["wg"] + self.B["bg"]
        return z, g

    def actions(self, psi, e):
        psi = torch.tensor(np.ascontiguousarray(psi), dtype=self.dtype).reshape(-1, AFEAT)
        pay = psi[:, AOFF_PAYS:AOFF_PAYS + NTYPE].unsqueeze(1) @ e
        return F.relu(torch.cat([psi, pay.squeeze(1)], -1) @ self.W["wq"] + self.B["bq"])

    # ------------------------------------------------------- phases

    def head(self, rows, xbel):
        """xbel [rows, 2dg] -> u [rows, rk]; ports Mlp::pbs_head."""
        out = xbel @ self.W["wb"]
        out = out + self.h0[:rows]
        out = F.relu(F.layer_norm(out + self.B["b1"], [self.hd],
                                  self.L["ln1w"], self.L["ln1b"]))
        return out @ self.W["wu"] + self.B["bu"]

    def belief_sums(self, traverser):
        """Phase 1: xb[leaf, player] = sum_c w_c z[c]. Ports leaf_values,
        including the alternating-traverser cache."""
        j = self.j
        redo = self.last_traverser
        self.last_traverser = traverser
        for p in (0, 1):
            if redo is not None and redo != p:
                continue
            for r, leaf in enumerate(j.leaf_rows):
                leaf = int(leaf)
                n = int(j.nc[leaf, p])
                w = self.normalize(self.reach_of(leaf, p))
                c0, c1 = j.leaf_coff[2 * r + p], j.leaf_coff[2 * r + p + 1]
                idx = torch.tensor(j.leaf_cidx[c0:c1], dtype=torch.long)
                z = self.z[idx]
                self.xb[r, p * self.dg:(p + 1) * self.dg] = (w[:, None] * z).sum(0)

    def belief_sums_both(self):
        """Both players' blocks, for the fixed-policy passes."""
        j = self.j
        self.last_traverser = None
        for p in (0, 1):
            for r, leaf in enumerate(j.leaf_rows):
                leaf = int(leaf)
                n = int(j.nc[leaf, p])
                w = self.normalize(self.reach_of(leaf, p))
                c0, c1 = j.leaf_coff[2 * r + p], j.leaf_coff[2 * r + p + 1]
                idx = torch.tensor(j.leaf_cidx[c0:c1], dtype=torch.long)
                z = self.z[idx]
                self.xb[r, p * self.dg:(p + 1) * self.dg] = (w[:, None] * z).sum(0)

    def readout(self, p):
        """Phase 3: per-leaf per-config values. Ports Solver::readout."""
        j = self.j
        opp = 1 - p
        u = self.head(j.rows, self.xb)
        for k, leaf in enumerate(j.term_leaves):
            leaf = int(leaf)
            u_term = float(j.terminal_utility[k])
            if j.node_player[leaf] != p:
                u_term = -u_term
            opp_reach = float(self.reach_of(leaf, opp).sum())
            n = int(j.nc[leaf, p])
            self.vals[j.voff[leaf]:j.voff[leaf] + n] = u_term * opp_reach
        for r, leaf in enumerate(j.leaf_rows):
            leaf = int(leaf)
            opp_reach = float(self.reach_of(leaf, opp).sum())
            c0, c1 = j.leaf_coff[2 * r + p], j.leaf_coff[2 * r + p + 1]
            idx = torch.tensor(j.leaf_cidx[c0:c1], dtype=torch.long)
            g = self.g[idx]
            v = (u[r] * g[:, :self.rk]).sum(1) + g[:, self.rk]
            n = int(j.nc[leaf, p])
            self.vals[j.voff[leaf]:j.voff[leaf] + n] = v * opp_reach

    def backprop(self, traverser, strat, mode):
        """Phase 4: bottom-up value propagation. Ports Solver::backprop.
        mode: 0 = regret, 1 = value, 2 = best response."""
        j = self.j
        ls = j.level_start
        for lev in range(len(ls) - 2, -1, -1):
            for i0 in j.bfs_order[ls[lev]:ls[lev + 1]]:
                i = int(i0)
                kind = int(j.node_kind[i])
                if kind == 2:
                    continue
                me = int(j.node_player[i])
                nc = int(j.nc[i, traverser])
                vbase = int(j.voff[i])
                if kind == 1:  # chance
                    ch = int(j.child[i][0])
                    # The child's full vals span: a draw child's support for
                    # the drawing player is the convolved (larger) set.
                    src = self.vals[j.voff[ch]:j.voff[ch + 1]]
                    if me == traverser:
                        d0, d1 = int(j.draw_off[i]), int(j.draw_off[i + 1])
                        b = j.draw_row_start[j.draw_row_off[i]:j.draw_row_off[i + 1]]
                        dst = torch.zeros(nc, dtype=self.dtype)
                        for c in range(nc):
                            to = j.draw_to[d0 + b[c]:d0 + b[c + 1]]
                            pr = torch.tensor(j.draw_p[d0 + b[c]:d0 + b[c + 1]],
                                              dtype=self.dtype)
                            dst[c] = (pr * src[to]).sum()
                        self.vals[vbase:vbase + nc] = dst
                    else:
                        self.vals[vbase:vbase + nc] = src[:nc]
                    continue
                # decision
                na = int(j.na[i])
                so = int(j.soff[i])
                cur = strat[so:so + nc * na].reshape(nc, na)
                if me == traverser:
                    if mode == 0:
                        self.inst[so:so + nc * na] = 0
                    vi = torch.full((nc,), float("-inf") if mode == 2 else 0.0,
                                    dtype=self.dtype)
                    for a in range(na):
                        ch = self.child_of(i, a)
                        cv = self.vals[j.voff[ch]:j.voff[ch + 1]]
                        col = self.trans[so + a:so + nc * na:na]
                        lcol = self.legal[so + a:so + nc * na:na]
                        ok = lcol & (col >= 0)
                        av = torch.zeros(nc, dtype=self.dtype)
                        av[ok] = cv[col[ok].clamp(min=0)]
                        if mode == 0:
                            self.inst[so + a:so + nc * na:na] += av
                            vi += av * cur[:, a]
                        elif mode == 1:
                            vi += av * cur[:, a]
                        else:
                            # Best response maxes over the *legal* actions
                            # only: an illegal cell must not pin the max at 0.
                            av = torch.where(ok, av, torch.full_like(av, float("-inf")))
                            vi = torch.maximum(vi, av)
                    if mode == 0:
                        for a in range(na):
                            lcol = self.legal[so + a:so + nc * na:na]
                            self.inst[so + a:so + nc * na:na] -= torch.where(
                                lcol, vi, torch.zeros_like(vi))
                    elif mode == 2:
                        vi = torch.where(vi == float("-inf"), torch.zeros_like(vi), vi)
                    self.vals[vbase:vbase + nc] = vi
                else:
                    vi = torch.zeros(nc, dtype=self.dtype)
                    for ch in self.child[i]:
                        vi = vi + self.vals[j.voff[int(ch)]:j.voff[int(ch) + 1]][:nc]
                    self.vals[vbase:vbase + nc] = vi

    def propagate(self, strat, root0, root1):
        """Phase 6: reach push-down. Ports Solver::propagate."""
        j = self.j
        self.reach.fill_(0)
        n0, n1 = int(j.nc[0, 0]), int(j.nc[0, 1])
        self.reach[:n0] = torch.tensor(root0, dtype=self.dtype)
        self.reach[n0:n0 + n1] = torch.tensor(root1, dtype=self.dtype)
        ls = j.level_start
        for lev in range(len(ls) - 1):
            for i0 in j.bfs_order[ls[lev]:ls[lev + 1]]:
                i = int(i0)
                kind = int(j.node_kind[i])
                if kind == 2:
                    continue
                me = int(j.node_player[i])
                op = 1 - me
                n_me, n_op = int(j.nc[i, me]), int(j.nc[i, op])
                base = int(j.reach_off[i])
                me_at = base + (int(j.nc[i, 0]) if me == 1 else 0)
                op_at = base + (int(j.nc[i, 0]) if op == 1 else 0)
                src_me = self.reach[me_at:me_at + n_me]
                if kind == 1:  # chance
                    ch = int(j.child[i][0])
                    cbase = int(j.reach_off[ch])
                    c_me_at = cbase + (int(j.nc[ch, 0]) if me == 1 else 0)
                    c_op_at = cbase + (int(j.nc[ch, 0]) if op == 1 else 0)
                    self.reach[c_op_at:c_op_at + n_op] = self.reach[op_at:op_at + n_op]
                    d0, d1 = int(j.draw_off[i]), int(j.draw_off[i + 1])
                    b = j.draw_row_start[j.draw_row_off[i]:j.draw_row_off[i + 1]]
                    for c in range(n_me):
                        to = j.draw_to[d0 + b[c]:d0 + b[c + 1]]
                        pr = torch.tensor(j.draw_p[d0 + b[c]:d0 + b[c + 1]],
                                          dtype=self.dtype)
                        self.reach[c_me_at + to] += src_me[c] * pr
                    continue
                na = int(j.na[i])
                so = int(j.soff[i])
                cur = strat[so:so + n_me * na].reshape(n_me, na)
                o = self.obs_of(i)
                act_base = int(j.action_off[i])
                for ch_i in range(len(self.child[i])):
                    ch = int(self.child[i][ch_i])
                    cbase = int(j.reach_off[ch])
                    c_me_at = cbase + (int(j.nc[ch, 0]) if me == 1 else 0)
                    c_op_at = cbase + (int(j.nc[ch, 0]) if op == 1 else 0)
                    self.reach[c_op_at:c_op_at + n_op] = self.reach[op_at:op_at + n_op]
                    for a in j.obs_act[act_base + o[ch_i]:act_base + o[ch_i + 1]]:
                        a = int(a)
                        col = self.trans[so + a:so + n_me * na:na]
                        lcol = self.legal[so + a:so + n_me * na:na]
                        ok = lcol & (col >= 0)
                        tgt = col[ok]
                        w = src_me[ok] * cur[ok, a]
                        self.reach[c_me_at + tgt] += w

    def normalize(self, w):
        tot = float(w.sum())
        if tot > SMOOTH:
            return w * (1.0 / tot)
        return torch.full_like(w, 1.0 / max(len(w), 1))

    def step(self, traverser):
        j = self.j
        # update_regrets: phases 1-4.
        self.belief_sums(traverser)
        self.readout(traverser)
        self.backprop(traverser, self.cur, 0)
        # RM block.
        m = self.steps[traverser] + 1.0
        da, db = self.factor(m, j.alpha), self.factor(m, j.beta)
        dg = (m / (m + 1.0)) ** j.gamma
        for i in range(j.nodes):
            if j.node_kind[i] != 0 or j.node_player[i] != traverser:
                continue
            na = int(j.na[i])
            nc = int(j.nc[i, traverser])
            so = int(j.soff[i])
            cells = slice(so, so + nc * na)
            leg = self.legal[cells].reshape(nc, na)
            reg = self.regret[cells].reshape(nc, na)
            ins = self.inst[cells].reshape(nc, na)
            r = reg * torch.where(reg > 0, da, db) + ins
            self.regret[cells] = r.reshape(-1)
            v = torch.clamp(r + j.predict * ins, min=EPS)
            v = torch.where(leg, v, torch.zeros_like(v))
            s = v.sum(1, keepdim=True)
            self.cur[cells] = torch.where(s > 0, v / s, v).reshape(-1)
            self.sum_strat[cells] *= dg
        # Forward reach sweep under the new strategy.
        self.propagate(self.cur, j.root0, j.root1)
        # AVG block.
        for i in range(j.nodes):
            if j.node_kind[i] != 0 or j.node_player[i] != traverser:
                continue
            na = int(j.na[i])
            nc = int(j.nc[i, traverser])
            so = int(j.soff[i])
            cells = slice(so, so + nc * na)
            r = self.reach_of(i, traverser)
            self.sum_strat[cells] += (r[:, None] * self.cur[cells].reshape(nc, na)).reshape(-1)
            ss = self.sum_strat[cells].reshape(nc, na)
            s = ss.sum(1, keepdim=True)
            leg = self.legal[cells].reshape(nc, na)
            unif = leg / leg.sum(1, keepdim=True).clamp(min=1)
            self.avg[cells] = torch.where(s > 0, ss / s, unif).reshape(-1)
        self.snapshot()
        self.steps[traverser] += 1

    @staticmethod
    def factor(t, p):
        if p == float("inf"):
            return 1.0
        if p == float("-inf"):
            return 0.0
        x = t ** p
        return x / (x + 1.0)

    def multistep(self, iters):
        for t in range(iters):
            self.step(t % 2)

    def value_under(self, roots):
        reference = self.snaps[-1]
        out = []
        for root in roots:
            self.propagate(reference, root[0], root[1])
            self.belief_sums_both()
            pair = []
            for p in (0, 1):
                self.readout(p)
                self.backprop(p, reference, 1)
                n = int(self.j.nc[0, p])
                pair.append(self.vals[self.j.voff[0]:self.j.voff[0] + n].clone())
            out.append(pair)
        return out

    def carried_beliefs(self, leaf):
        j = self.j
        out = []
        for snap in self.snaps[:-1]:
            self.propagate(snap, j.root0, j.root1)
            pair = [self.normalize(self.reach_of(leaf, p)).clone() for p in (0, 1)]
            out.append(pair)
        return out

    def nash_conv(self):
        j = self.j
        reference = self.snaps[-1]
        self.propagate(reference, j.root0, j.root1)
        self.belief_sums_both()
        nash, zero = 0.0, 0.0
        for p in (0, 1):
            self.readout(p)
            nc = int(j.nc[0, p])
            vo = int(j.voff[0])
            root = j.root0 if p == 0 else j.root1
            expect = lambda v: sum(float(root[c]) * float(v[c]) for c in range(nc))
            self.backprop(p, reference, 1)
            v = expect(self.vals[vo:vo + nc])
            self.backprop(p, reference, 2)
            nash += expect(self.vals[vo:vo + nc]) - v
            zero += v
        return nash, zero


    # ------------------------------------------------------- warm start

    def policy_into_cur(self):
        """The policy head's distribution into `cur` at every decision node.
        Ports Solver::policy_into_cur (A4 warm start): q(a) from the action
        features, u_pi from the shared hidden layer, k(c) from the config
        tower, softmax over the legal actions with the 1e-6 floor."""
        j = self.j
        if j.warm <= 0 or j.n_inner == 0:
            return False
        base = j.nleaf
        for k, i in enumerate(j.inner_rows):
            i = int(i)
            r = base + k
            me = int(j.node_player[i])
            na = int(j.na[i])
            nc = int(j.nc[i, me])
            so = int(j.soff[i])
            xbel = torch.zeros(2 * self.dg, dtype=self.dtype)
            for p in (0, 1):
                w = self.normalize(self.reach_of(i, p))
                c0, c1 = j.leaf_coff[2 * r + p], j.leaf_coff[2 * r + p + 1]
                idx = torch.tensor(j.leaf_cidx[c0:c1], dtype=torch.long)
                xbel[p * self.dg:(p + 1) * self.dg] = (w[:, None] * self.z[idx]).sum(0)
            q = self.q[i]
            out = xbel @ self.W["wb"]
            out = out + self.h0[r]
            hid = F.relu(F.layer_norm(out + self.B["b1"], [self.hd],
                                      self.L["ln1w"], self.L["ln1b"]))
            upi = hid @ self.W["wp"] + self.B["bp"]
            c0, c1 = j.leaf_coff[2 * r + me], j.leaf_coff[2 * r + me + 1]
            idx = torch.tensor(j.leaf_cidx[c0:c1], dtype=torch.long)
            kk = self.z[idx] @ self.W["wk"] + self.B["bk"] + upi  # [nc, rk]
            logit = kk @ q.t()                                     # [nc, na]
            leg = self.legal[so:so + nc * na].reshape(nc, na)
            m = torch.where(leg, logit, torch.full_like(logit, float("-inf"))).max(1,
                                                                                  keepdim=True).values
            v = torch.where(leg, torch.exp(logit - m), torch.zeros_like(logit))
            s = v.sum(1, keepdim=True)
            cur = torch.where(s > 0, v / s, v)
            cur = torch.clamp(cur, min=0.0)
            cur = torch.where(leg, torch.clamp(cur, min=EPS), torch.zeros_like(cur))
            self.cur[so:so + nc * na] = cur.reshape(-1)
        return True

    def warm_start(self, weight):
        """Seed CFR from the policy head. Ports Solver::warm_start."""
        j = self.j
        if weight <= 0:
            return
        if not self.policy_into_cur():
            return
        self.propagate(self.cur, j.root0, j.root1)
        for p in (0, 1):
            self.belief_sums(p)
            self.readout(p)
            self.backprop(p, self.cur, 0)
            for i in range(j.nodes):
                if j.node_kind[i] != 0 or j.node_player[i] != p:
                    continue
                na = int(j.na[i])
                nc = int(j.nc[i, p])
                so = int(j.soff[i])
                cells = slice(so, so + nc * na)
                self.regret[cells] = weight * self.inst[cells]
                r = self.reach_of(i, p)
                self.sum_strat[cells] = (
                    weight * r[:, None] * self.cur[cells].reshape(nc, na)).reshape(-1)
            self.steps[p] = int(weight)
        # The t=0 snapshot was the uniform average; retake it from the seed.
        self.avg.zero_()
        for i in range(j.nodes):
            if j.node_kind[i] != 0:
                continue
            me = int(j.node_player[i])
            na = int(j.na[i])
            nc = int(j.nc[i, me])
            so = int(j.soff[i])
            cells = slice(so, so + nc * na)
            ss = self.sum_strat[cells].reshape(nc, na)
            s = ss.sum(1, keepdim=True)
            leg = self.legal[cells].reshape(nc, na)
            unif = leg / leg.sum(1, keepdim=True).clamp(min=1)
            self.avg[cells] = torch.where(s > 0, ss / s, unif).reshape(-1)
        self.snaps.clear()
        self.snap_t = 0
        self.snapshot()

# ------------------------------------------------------------------ check


def main():
    import glob
    import os
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--check":
        d = args[1]
        which = [int(x) for x in args[2:]] or None
        dims, W, B, L = load_weights(f"{d}/weights.bin")
        for path in sorted(glob.glob(f"{d}/solve_*.bin")):
            k = int(os.path.basename(path).split("_")[1].split(".")[0])
            if which is not None and k not in which:
                continue
            job = Job(path)
            sv = Solve(job, dims, W, B, L)
            sv.warm_start(job.warm)
            sv.multistep(job.iters)
            roots = job.carried or [[job.root0, job.root1]]
            vals = sv.value_under(roots)
            leaf = int(job.leaf_rows[0])
            carried = sv.carried_beliefs(leaf)
            nash, zero = sv.nash_conv()
            ref = Ref(f"{d}/solve_{k}.ref", job)
            errs = []
            def cmp(name, a, b, rtol=2e-4, atol=1e-4):
                # Scale-relative: with trained weights every value is ~+-1 and
                # atol dominates; the oracle dumps use random weights whose
                # values reach ~1000, where float32 accumulation is ~1e-3.
                a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
                if a.shape != b.shape:
                    errs.append(f"{name}: shape {a.shape} != {b.shape}")
                    return
                if a.size == 0:
                    return
                scale = max(np.abs(a).max(), np.abs(b).max())
                d = np.max(np.abs(a - b))
                if d > atol + rtol * scale:
                    errs.append(f"{name}: max diff {d:.3e} > {atol}+{rtol}*{scale:.3e}")
            # Strategies accumulate torch-vs-cblas rounding through the CFR
            # normalisations (~8e-4 at T=8 with random weights); the spec is a
            # logic check, so its strategy bound is looser than the GPU tests'.
            cmp("reference", sv.snaps[-1], ref.reference, rtol=1e-2)
            for t, (a, b) in enumerate(zip(sv.snaps, ref.snaps)):
                cmp(f"snap[{t}]", a, b, rtol=1e-2)
            for r, (a, b) in enumerate(zip(vals, ref.root_values)):
                cmp(f"root[{r}].0", a[0], b[0])
                cmp(f"root[{r}].1", a[1], b[1])
            for t, (a, b) in enumerate(zip(carried, ref.carried)):
                cmp(f"carried[{t}]", a[0], b[0])
                cmp(f"carried[{t}]", a[1], b[1])
            cmp("nash", [nash], [ref.nash], rtol=5e-4, atol=1e-3)
            cmp("zero", [zero], [ref.zero_sum], rtol=5e-4, atol=1e-3)
            status = "OK" if not errs else "FAIL"
            print(f"solve {k}: {status}" + ("" if not errs else "\n  " + "\n  ".join(errs)))
            if errs:
                sys.exit(1)
        print("all solves match the CPU reference")
    else:
        print(__doc__)


class Ref:
    """The oracle_dump reference outputs."""

    def __init__(self, path, job):
        b = open(path, "rb").read()
        at = [0]

        def take(n):
            s = b[at[0]:at[0] + n]
            assert len(s) == n
            at[0] += n
            return s

        def u32():
            return struct.unpack("<I", take(4))[0]

        def raw(n):
            return np.frombuffer(take(4 * n), np.float32)

        assert u32() == 0x57435252
        nsnaps, ncells = u32(), u32()
        self.reference = raw(ncells)
        self.snaps = [raw(ncells) for _ in range(nsnaps)]
        n0, n1 = int(job.nc[0, 0]), int(job.nc[0, 1])
        self.root_values = []
        for _ in range(u32()):
            self.root_values.append([raw(n0), raw(n1)])
        self.leaf = u32()
        # The carried beliefs live at the exit leaf, whose support differs
        # from the root's.
        ln0, ln1 = int(job.nc[self.leaf, 0]), int(job.nc[self.leaf, 1])
        self.carried = []
        for _ in range(u32()):
            self.carried.append([raw(ln0), raw(ln1)])
        assert u32() == 1
        self.nash = float(raw(1)[0])
        assert u32() == 1
        self.zero_sum = float(raw(1)[0])
        assert at[0] == len(b)


if __name__ == "__main__":
    main()
