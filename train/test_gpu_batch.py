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
from gpu_batch import make_batch as make_gpu_batch, warmup
from train import make_batch as make_cpu_batch


def clone(parts):
    return tuple(tuple(np.array(y, copy=True) for y in x)
                 if isinstance(x, tuple) else np.array(x, copy=True)
                 for x in parts)


def empty_policy():
    return (np.zeros((0, warchest.ACT_BYTES), np.uint8),
            np.zeros(0, np.int64), np.zeros(0, np.int64),
            np.zeros(0, np.int64), np.zeros(0, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=1024)
    args = ap.parse_args()

    dev = torch.device(args.device)
    d = Dump(args.dump)
    parts = (*d.rows(0, min(args.rows, len(d))), empty_policy())
    warmup(dev)
    a = make_cpu_batch(clone(parts), np.random.default_rng(17), dev)
    b = make_gpu_batch(clone(parts), np.random.default_rng(17), dev)
    torch.cuda.synchronize(dev)
    for x, y in zip(a[:5], b[:5]):
        if x.dtype.is_floating_point:
            torch.testing.assert_close(x, y, rtol=0, atol=1e-6)
        else:
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert a[5] == b[5]
    for x, y in zip(a[6], b[6]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    print(f"{len(parts[0])} rows, {len(parts[1])} configs OK")


if __name__ == "__main__":
    main()
