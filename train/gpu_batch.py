"""Build training batches and expand compact rows with the CUDA kernel."""

from functools import lru_cache

import numpy as np
import torch

import mirror
import warchest
from value_net import AFEAT

N_KINDS = warchest.N_KINDS
NSLOT = warchest.NSLOT
N_HEXES = warchest.N_HEXES


@lru_cache(maxsize=None)
def _tables(device_text):
    device = torch.device(device_text)
    cards = torch.as_tensor(
        np.asarray(warchest.card_features_table(), np.float32), device=device)
    locations = torch.as_tensor(
        np.asarray(warchest.hex_location_flags(), np.uint8), device=device)
    return cards, locations


def action_feats(pa):
    """Encode stored action bytes where the policy head will read them."""
    feat = torch.zeros((len(pa), AFEAT), dtype=torch.float32, device=pa.device)
    if not len(pa):
        return feat
    idx = torch.arange(len(pa), device=pa.device)
    feat[idx, pa[:, 0].long()] = 1.0
    feat[idx, N_KINDS + pa[:, 1].long()] = 1.0
    at = N_KINDS + NSLOT + 1
    for k in range(3):
        h = torch.where(pa[:, 2 + k] == 255, N_HEXES,
                        pa[:, 2 + k].long())
        feat[idx, at + h] = 1.0
        at += N_HEXES + 1
    return feat


def expand_rows(rows):
    """Expand original rows and their mirrored views on the CUDA device."""
    if rows.device.type != "cuda":
        raise ValueError("CUDA expansion requires CUDA rows")
    n = len(rows)
    views = torch.stack([rows, mirror.mirror_torch(rows)], 1).reshape(2 * n, -1)
    cards, locations = _tables(str(rows.device))
    x = torch.empty((2 * n, warchest.PUBFEAT), dtype=torch.float32,
                    device=rows.device)
    stream = torch.cuda.current_stream(rows.device)
    ordinal = (rows.device.index
               if rows.device.index is not None
               else torch.cuda.current_device())
    warchest.expand_rows_cuda(
        views.data_ptr(), cards.data_ptr(), locations.data_ptr(), x.data_ptr(),
        2 * n, stream.cuda_stream, ordinal)
    return x


def make_batch(parts, device):
    """Make the one canonical training batch from gathered replay payload."""
    rows, cc, _cp, cw, cy, seg, pol = parts
    device = torch.device(device)

    def tensor(value, dtype):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=device)

    rows = tensor(rows, torch.uint8)
    n = len(rows)
    if device.type == "cuda":
        x = expand_rows(rows)
    else:
        raw = rows.numpy()
        views = np.empty((2 * n, warchest.ROW_BYTES), np.uint8)
        views[0::2] = raw
        views[1::2] = mirror.mirror_rows(raw)
        x = torch.as_tensor(
            np.asarray(warchest.expand_rows(views.ravel()), np.float32)
            .reshape(2 * n, -1),
            device=device)

    phi = tensor(cc, torch.float32) / float(warchest.CNORM)
    pa, pact, parow, pcfg, group, pprob, group_count = pol
    policy = (
        action_feats(tensor(pa, torch.uint8)),
        tensor(parow, torch.long),
        tensor(pact, torch.long),
        tensor(pcfg, torch.long),
        tensor(group, torch.long),
        tensor(pprob, torch.float32),
        int(group_count),
    )
    return (x, phi, tensor(cw, torch.float32), tensor(seg, torch.long),
            tensor(cy, torch.float32), 2 * n, policy)


def warmup(device):
    """Compile the expansion kernel before the run's wall-clock starts."""
    rows = torch.zeros((1, warchest.ROW_BYTES), dtype=torch.uint8, device=device)
    rows[:, warchest.ROW_HEX_OWNER:warchest.ROW_HEX_OWNER + warchest.N_HEXES] = 255
    rows[:, warchest.ROW_HEX_SLOT:warchest.ROW_HEX_SLOT + warchest.N_HEXES] = 255
    rows[:, warchest.ROW_HEX_MARKER:warchest.ROW_HEX_MARKER + warchest.N_HEXES] = 255
    cc = torch.zeros((2, warchest.CCOUNTS), dtype=torch.uint8, device=device)
    # An empty policy: a row without a target is exactly what the warm start
    # and every query solve look like, so this is the shape, not a special case.
    empty = (
        torch.zeros((0, warchest.ACT_BYTES), dtype=torch.uint8, device=device),
        torch.zeros(0, dtype=torch.int16, device=device),
        torch.zeros(0, dtype=torch.int64, device=device),
        torch.zeros(0, dtype=torch.int64, device=device),
        torch.zeros(0, dtype=torch.int64, device=device),
        torch.zeros(0, dtype=torch.float32, device=device),
        0,
    )
    parts = (rows, cc,
             torch.as_tensor([0, 1], dtype=torch.uint8, device=device),
             torch.as_tensor([1.0, 1.0], device=device),
             torch.zeros(2, device=device),
             torch.as_tensor([0, 1], dtype=torch.int64, device=device), empty)
    make_batch(parts, device)
    torch.cuda.synchronize(device)
