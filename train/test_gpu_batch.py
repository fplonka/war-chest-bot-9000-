"""Parity gate for the CUDA replay expander against the Rust encoder.

Run on a CUDA box with any frozen replay dump::

    python train/test_gpu_batch.py runs/pre_cuda_random/buffer.npz --device cuda:0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import warchest

from dump import Dump
from gpu_batch import expand_rows, warmup
import mirror


def empty_policy():
    return (np.zeros((0, warchest.ACT_BYTES), np.uint8),
            np.zeros(0, np.int16), np.zeros(0, np.int64),
            np.zeros(0, np.int64), np.zeros(0, np.int64),
            np.zeros(0, np.float32), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=1024)
    args = ap.parse_args()

    dev = torch.device(args.device)
    d = Dump(args.dump)
    parts = (*d.rows(0, min(args.rows, len(d))), empty_policy())
    rows = parts[0]
    views = np.empty((2 * len(rows), warchest.ROW_BYTES), np.uint8)
    views[0::2] = rows
    views[1::2] = mirror.mirror_rows(rows)
    cpu = np.asarray(warchest.expand_rows(views.ravel()), np.float32)
    cpu = cpu.reshape(2 * len(rows), -1)
    warmup(dev)
    gpu = expand_rows(torch.as_tensor(rows, dtype=torch.uint8, device=dev))
    torch.cuda.synchronize(dev)
    torch.testing.assert_close(
        torch.as_tensor(cpu, device=dev), gpu, rtol=0, atol=1e-6)
    print(f"{len(parts[0])} rows, {len(parts[1])} configs OK")


if __name__ == "__main__":
    main()
