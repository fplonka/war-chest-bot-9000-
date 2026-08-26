"""Expand compact replay rows with the engine's CUDA kernel."""

from functools import lru_cache

import numpy as np
import torch

import mirror
import warchest
from train import action_feats


@lru_cache(maxsize=None)
def _tables(device_text):
    device = torch.device(device_text)
    cards = torch.as_tensor(
        np.asarray(warchest.card_features_table(), np.float32), device=device)
    locations = torch.as_tensor(
        np.asarray(warchest.hex_location_flags(), np.uint8), device=device)
    return cards, locations


def make_batch(parts, rng, device):
    """Device replay batch -> two canonical query rows on ``device``.

    ``parts`` comes from ``Buffer.gather`` and is already on the device; this
    mirrors the rows, runs the engine's expander, and assembles the network
    input. Nothing crosses the host.
    """
    del rng
    rows, cc, cp, cw, cy, seg, pol = parts
    n = len(rows)
    views = torch.stack([rows, mirror.mirror_torch(rows)], 1).reshape(2 * n, -1)
    cards, locations = _tables(str(device))
    x = torch.empty((2 * n, warchest.PUBFEAT), dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream(device)
    ordinal = device.index if device.index is not None else torch.cuda.current_device()
    warchest.expand_rows_cuda(
        views.data_ptr(), cards.data_ptr(), locations.data_ptr(), x.data_ptr(),
        2 * n, stream.cuda_stream, ordinal)
    phi = cc.to(torch.float32) / float(warchest.CNORM)
    pa, pact, pcrow, pcfg, pprob, parow = pol
    policy = (action_feats(pa), parow.long(), pact.long(), pcrow.long(),
              pcfg.long(), pprob.to(torch.float32))
    return (x, phi, cw.to(torch.float32), seg, cy.to(torch.float32), 2 * n,
            policy)


def warmup(device):
    """Compile the expansion kernel before the run's wall-clock starts."""
    rows = torch.zeros((1, warchest.ROW_BYTES), dtype=torch.uint8, device=device)
    rows[:, warchest.ROW_HEX_OWNER:warchest.ROW_HEX_OWNER + warchest.N_HEXES] = 255
    rows[:, warchest.ROW_HEX_SLOT:warchest.ROW_HEX_SLOT + warchest.N_HEXES] = 255
    rows[:, warchest.ROW_HEX_MARKER:warchest.ROW_HEX_MARKER + warchest.N_HEXES] = 255
    cc = torch.zeros((2, warchest.CCOUNTS), dtype=torch.uint8, device=device)
    # An empty policy: a row without a target is exactly what the warm start
    # and every query solve look like, so this is the shape, not a special case.
    empty = (torch.zeros((0, warchest.ACT_BYTES), dtype=torch.uint8, device=device),
             torch.zeros(0, dtype=torch.int64, device=device),
             torch.zeros(0, dtype=torch.int64, device=device),
             torch.zeros(0, dtype=torch.int64, device=device),
             torch.zeros(0, dtype=torch.float32, device=device),
             torch.zeros(0, dtype=torch.int64, device=device))
    parts = (rows, cc, torch.as_tensor([0, 1], torch.uint8, device=device),
             torch.as_tensor([1.0, 1.0], device=device),
             torch.zeros(2, device=device),
             torch.as_tensor([0, 1], torch.int64, device=device), empty)
    make_batch(parts, np.random.default_rng(0), device)
    torch.cuda.synchronize(device)
