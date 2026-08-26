"""The device-backed replay ring used by the trainer."""

import time

import numpy as np
import torch

import warchest


CCOUNTS = warchest.CCOUNTS
ROW_BYTES = warchest.ROW_BYTES
ACT_BYTES = warchest.ACT_BYTES


def _to_numpy(value):
    if isinstance(value, tuple):
        return tuple(_to_numpy(item) for item in value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _policy_group_counts(pcoff, pci, rows):
    counts = np.zeros(rows, np.int32)
    for row, (lo, hi) in enumerate(zip(pcoff[:-1], pcoff[1:])):
        counts[row] = np.unique(pci[lo:hi]).size
    return counts


class Buffer:
    """Fixed-capacity FIFO over rows with ragged config and policy arenas.

    Payload rings live on the training device: packed rows, config counts,
    player labels, weights, targets, actions, policy cells, and probabilities.
    Index and metadata rings stay on the host: arena starts and lengths,
    timestamps, source labels, truth, outcomes, and TD(1) labels.

    Append retires the oldest rows until the new chunk fits every ring. Absolute
    arena offsets make each live row's spans unambiguous even after wraparound.
    Device payloads are preallocated at the configured capacity. Host metadata
    is kept in compact NumPy arrays for eviction, sampling diagnostics, and the
    host-side arena arithmetic.
    """

    def __init__(self, cap, ccap, device):
        self.cap, self.ccap = cap, ccap
        self.device = torch.device(device)
        self.x = torch.zeros((cap, ROW_BYTES), dtype=torch.uint8,
                             device=self.device)
        self.soff = np.zeros(0, np.int64)

        self.cstart = np.zeros(cap, np.int64)
        self.clen = np.zeros((cap, 2), np.int32)
        self.cc = torch.zeros((ccap, CCOUNTS), dtype=torch.uint8,
                              device=self.device)
        self.cp = torch.zeros(ccap, dtype=torch.uint8, device=self.device)
        self.cw = torch.zeros(ccap, dtype=torch.float16, device=self.device)
        self.cy = torch.zeros(ccap, dtype=torch.float16, device=self.device)

        self.pastart = np.zeros(cap, np.int64)
        self.palen = np.zeros(cap, np.int32)
        self.pcstart = np.zeros(cap, np.int64)
        self.pclen = np.zeros(cap, np.int32)
        self.pgroup_count = np.zeros(cap, np.int32)
        self.acap = cap * 24
        self.pcap = cap * 96
        self.pa = torch.zeros((self.acap, ACT_BYTES), dtype=torch.uint8,
                              device=self.device)
        self.pci = torch.zeros(self.pcap, dtype=torch.int32,
                               device=self.device)
        self.pact = torch.zeros(self.pcap, dtype=torch.int16,
                                device=self.device)
        self.pp = torch.zeros(self.pcap, dtype=torch.float16,
                              device=self.device)

        self.written_at = np.zeros(cap, np.float64)
        self.created_at = np.zeros(cap, np.float64)
        self.source = np.zeros(cap, np.uint8)  # 0 warm, 1 play, 2 query
        self.truth = np.zeros((cap, 2), np.uint32)
        self.outcome = np.full((cap, 2), np.nan, np.float32)
        self.td1 = np.zeros(cap, np.uint8)

        self.acts = 0
        self.cells = 0
        self.rows = 0
        self.cfgs = 0
        self.lo = 0

    def add(self, x, cc, cw, cy, coff, soff, source, truth, outcome, created,
            td1, pol=None):
        n = len(x)
        if not x.flags.writeable:
            x = x.copy()
        lens = np.diff(coff).reshape(n, 2)
        m = len(cw)
        if pol is None:
            na = nc = 0
            group_counts = np.zeros(n, np.int32)
        else:
            pa, paoff, pcoff, pci, pact, pprob = pol
            na, nc = len(pa), len(pci)
            group_counts = _policy_group_counts(pcoff, pci, n)

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
            ring = sl % self.cap
            self.x[ring] = torch.as_tensor(x[i:j], device=self.device)
            self.cstart[ring] = starts[i:j]
            self.clen[ring] = lens[i:j]
            self.pgroup_count[ring] = group_counts[i:j]
            self.written_at[ring] = now
            self.created_at[ring] = created[i:j]
            self.source[ring] = source[i:j]
            self.truth[ring] = truth[i:j]
            self.outcome[ring] = outcome[i:j]
            self.td1[ring] = td1[i:j]

        sl = (np.arange(m) + self.cfgs) % self.ccap
        self.cc[sl] = torch.as_tensor(cc, device=self.device)
        self.cp[sl] = torch.as_tensor(cp, device=self.device)
        self.cw[sl] = torch.as_tensor(cw, dtype=torch.float16,
                                       device=self.device)
        self.cy[sl] = torch.as_tensor(cy, dtype=torch.float16,
                                      device=self.device)

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
            self.pa[(np.arange(na) + self.acts) % self.acap] = \
                torch.as_tensor(pa, device=self.device)
            at = (np.arange(nc) + self.cells) % self.pcap
            self.pci[at] = torch.as_tensor(pci, dtype=torch.int32,
                                           device=self.device)
            self.pact[at] = torch.as_tensor(pact, dtype=torch.int16,
                                            device=self.device)
            self.pp[at] = torch.as_tensor(pprob, dtype=torch.float16,
                                          device=self.device)
            self.acts += na
            self.cells += nc

        self.rows += n
        self.cfgs += m
        self.soff = np.concatenate(
            [self.soff, np.asarray(soff, np.int64)[1:] + base])
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

    def _gather(self, ids):
        ids = np.asarray(ids, np.int64)
        s = ids % self.cap
        lens = self.clen[s].sum(1).astype(np.int64)
        alen = self.palen[s].astype(np.int64)
        clen = self.pclen[s].astype(np.int64)
        n = len(ids)
        dev = self.device
        t = lambda a: torch.as_tensor(a, dtype=torch.long, device=dev)
        s_t, lens_t, alen_t, clen_t = t(s), t(lens), t(alen), t(clen)
        row_ids = torch.arange(n, dtype=torch.long, device=dev)
        total = int(lens.sum())
        alen_total = int(alen.sum())
        clen_total = int(clen.sum())

        rowstart = torch.cumsum(lens_t, 0) - lens_t
        cstart = t(self.cstart[s])
        at = (torch.repeat_interleave(cstart - rowstart, lens_t,
                                      output_size=total)
              + torch.arange(total, device=dev)) % self.ccap
        seg = 2 * torch.repeat_interleave(row_ids, lens_t,
                                          output_size=total) + self.cp[at]

        astart = torch.cumsum(alen_t, 0) - alen_t
        pstart = t(self.pastart[s])
        ccell_start = t(self.pcstart[s])
        ai = (torch.repeat_interleave(pstart - astart, alen_t,
                                      output_size=alen_total)
              + torch.arange(alen_total, device=dev))
        ci = (torch.repeat_interleave(ccell_start
                                      - torch.cumsum(clen_t, 0) + clen_t,
                                      clen_t, output_size=clen_total)
              + torch.arange(clen_total, device=dev))
        pcfg = (torch.repeat_interleave(rowstart, clen_t,
                                        output_size=clen_total)
                + self.pci[ci % self.pcap])
        if clen_total:
            group = torch.empty(clen_total, dtype=torch.long, device=dev)
            group[0] = 0
            group[1:] = (pcfg[1:] != pcfg[:-1]).cumsum(0)
        else:
            group = torch.empty(0, dtype=torch.long, device=dev)
        policy = (
            self.pa[ai % self.acap],
            self.pact[ci % self.pcap],
            torch.repeat_interleave(row_ids, alen_t,
                                    output_size=alen_total),
            pcfg,
            group,
            self.pp[ci % self.pcap],
            int(self.pgroup_count[s].sum(dtype=np.int64)),
        )
        cw = self.cw[at].to(torch.float32)
        mass = torch.zeros(2 * n, device=dev).index_add_(0, seg, cw)
        return (self.x[s_t], self.cc[at], self.cp[at], cw / mass[seg],
                self.cy[at], seg, policy)

    def gather(self, ids):
        """Assemble raw payload and index tensors for one training batch."""
        return self._gather(ids)

    def sample_ids(self, batch, rng, recent_mix=0.0, recent_frac=0.2):
        """Return absolute row ids sampled from the live replay."""
        ids = rng.integers(self.lo, self.rows, size=batch)
        k = int(batch * recent_mix)
        if k:
            span = max(1, int((self.rows - self.lo) * recent_frac))
            ids[:k] = rng.integers(self.rows - span, self.rows, size=k)
        return ids

    def sample(self, batch, rng, recent_mix=0.0, recent_frac=0.2):
        return self.gather(self.sample_ids(batch, rng, recent_mix, recent_frac))

    def sample_old(self, batch, rng, recent_frac=0.2):
        """Sample from outside the recent slice for the age diagnostic."""
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
            "replay_td1_target_frac": 2.0 * self.td1[ids].sum()
            / max(configs, 1),
            "target_age_max": (time.time() - self.created_at[ids].min()
                               if len(ids) else 0.0),
        }

    def sample_calibration(self, batch, rng):
        """Rows with game outcomes and their true-config indices."""
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
        """Return live raw payload oldest-first, as host arrays."""
        return _to_numpy(self._gather(np.arange(self.lo, self.rows)))
