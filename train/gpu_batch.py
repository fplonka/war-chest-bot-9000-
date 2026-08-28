"""Expand compact replay rows with the engine's CUDA kernel."""

from functools import lru_cache

import numpy as np
import torch

import mirror
import warchest


@lru_cache(maxsize=None)
def _tables(device_text):
    device = torch.device(device_text)
    cards = torch.as_tensor(
        np.asarray(warchest.card_features_table(), np.float32), device=device)
    locations = torch.as_tensor(
        np.asarray(warchest.hex_location_flags(), np.uint8), device=device)
    return cards, locations


def make_batch(parts, rng, device):
    """Compact replay batch -> two player queries on ``device``."""
    rows, cc, cw, cy, seg, pol = parts
    rows, seg, pol = mirror.mirror_batch(rows, seg, pol, rng.random(len(rows)) < 0.5)
    n = len(rows)
    rows = np.ascontiguousarray(rows)
    cc = np.ascontiguousarray(cc)
    t = lambda a, dtype=None: torch.as_tensor(a, dtype=dtype, device=device)
    rows_t = t(rows, torch.uint8)
    cards, locations = _tables(str(device))
    x = torch.empty((n, warchest.PUBFEAT), dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream(device)
    ordinal = device.index if device.index is not None else torch.cuda.current_device()
    warchest.expand_rows_cuda(
        rows_t.data_ptr(), cards.data_ptr(), locations.data_ptr(), x.data_ptr(),
        n, stream.cuda_stream, ordinal)

    phi = t(cc, torch.float32) / float(warchest.CNORM)
    pa, parow, pact, pcfg, pprob = pol
    policy = (t(pa, torch.uint8), t(parow, torch.long),
              t(pact, torch.long), t(pcfg, torch.long), t(pprob, torch.float32))
    return (x, phi, t(cw, torch.float32), t(seg, torch.long),
            t(cy, torch.float32), 2 * n, policy)


def warmup(device):
    """Compile the expansion kernel before the run's wall-clock starts."""
    rows = np.zeros((1, warchest.ROW_BYTES), np.uint8)
    rows[:, warchest.ROW_HEX_OWNER:warchest.ROW_HEX_OWNER + warchest.N_HEXES] = 255
    rows[:, warchest.ROW_HEX_SLOT:warchest.ROW_HEX_SLOT + warchest.N_HEXES] = 255
    rows[:, warchest.ROW_HEX_MARKER:warchest.ROW_HEX_MARKER + warchest.N_HEXES] = 255
    cc = np.zeros((2, warchest.CCOUNTS), np.uint8)
    # An empty policy: a row without a target is exactly what the warm start
    # and every query solve look like, so this is the shape, not a special case.
    empty = (np.zeros((0, warchest.ACT_BYTES), np.uint8),
             np.zeros(0, np.int64), np.zeros(0, np.int64),
             np.zeros(0, np.int64), np.zeros(0, np.float32))
    parts = (rows, cc, np.asarray([1.0, 1.0], np.float32),
             np.zeros(2, np.float32), np.asarray([0, 1], np.int64), empty)
    make_batch(parts, np.random.default_rng(0), device)
    torch.cuda.synchronize(device)
