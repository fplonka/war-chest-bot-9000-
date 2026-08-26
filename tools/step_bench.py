#!/usr/bin/env python3
"""Where does one optimizer step's GPU time go?

Times each stage of the value-and-policy step at the production batch on an
idle GPU, with CUDA events so the host never paces the device: the expander,
the physical trunk, the config encoder, the join, the policy head, the
backward pass, and Adam. A hundred back-to-back iterations, one sync at the
end, like a training burst with the synchronizations removed.

Also checks that the device batcher and the host oracle agree on the same
numpy parts (the gpu_batch parity gate), and what torch.compile does to the
forward.

    python tools/step_bench.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))

import numpy as np
import torch

import warchest
from train import ACT_BYTES, CCOUNTS, ROW_BYTES, Buffer, losses, make_batch as make_cpu_batch
from gpu_batch import make_batch as make_gpu_batch, warmup
from value_net import Net


def synth_parts(n, ncfg, na, ncells, seed=0):
    r = np.random.default_rng(seed)
    rows = r.integers(0, 256, size=(n, ROW_BYTES), dtype=np.uint8)
    cc = r.integers(0, 9, size=(n * ncfg, CCOUNTS), dtype=np.uint8)
    cw = np.full(n * ncfg, 1.0 / ncfg, np.float32)
    cy = r.uniform(-1, 1, size=n * ncfg).astype(np.float32)
    per = ncfg // 2
    coff = [0]
    for _ in range(n):
        coff.append(coff[-1] + per)
        coff.append(coff[-1] + (ncfg - per))
    coff = np.asarray(coff, np.int64)
    soff = np.arange(n + 1, dtype=np.int64)
    pa = r.integers(0, 255, size=(n * na, ACT_BYTES), dtype=np.uint8)
    pa[:, 0] = r.integers(0, warchest.N_KINDS, size=n * na)
    pa[:, 1] = r.integers(0, warchest.NSLOT + 1, size=n * na)
    pa[:, 2:] = r.integers(0, warchest.N_HEXES + 1, size=(n * na, ACT_BYTES - 2))
    paoff = np.arange(n + 1, dtype=np.int64) * na
    pcoff = np.arange(n + 1, dtype=np.int64) * ncells
    pci = np.tile(np.arange(ncells) % ncfg, n).astype(np.uint16)
    pact = np.tile(np.arange(ncells) % na, n).astype(np.uint16)
    pprob = np.full(n * ncells, 1.0 / ncells, np.float32)
    source = np.ones(n, np.uint8)
    truth = np.zeros((n, 2), np.uint32)
    outcome = np.full((n, 2), np.nan, np.float32)
    created = np.zeros(n, np.float64)
    td1 = np.zeros(n, np.uint8)
    pol = (pa, paoff, pcoff, pci, pact, pprob)
    return rows, cc, cw, cy, coff, soff, source, truth, outcome, created, td1, pol


def main():
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    n, ncfg, na, ncells = 256, 48, 24, 24
    buf = Buffer(n * 4, n * 4 * ncfg, dev)
    buf.add(*synth_parts(n, ncfg, na, ncells))
    ids = np.arange(buf.lo, buf.rows)
    device_parts = buf.gather(ids)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    warmup(dev)

    batch = make_gpu_batch(device_parts, np.random.default_rng(0), dev)
    xpub, phi, w, seg, y, nseg, policy = batch
    print(f"batch: xpub={tuple(xpub.shape)} phi={tuple(phi.shape)} "
          f"cells={tuple(policy[0].shape)} nseg={nseg}")

    # host oracle parity on the same data
    def to_numpy(value):
        if isinstance(value, tuple):
            return tuple(to_numpy(item) for item in value)
        return value.cpu().numpy() if torch.is_tensor(value) else value

    np_parts = to_numpy(device_parts)
    a = make_cpu_batch(np_parts, np.random.default_rng(0), dev)
    for i, (u, v) in enumerate(zip(a, batch)):
        if i == 5:
            assert u == v
        elif i == 6:
            for left, right in zip(u, v):
                torch.testing.assert_close(left, right, rtol=0, atol=1e-6)
        else:
            torch.testing.assert_close(u, v, rtol=0, atol=1e-6)
    print("gpu batcher == host oracle on the same parts: OK")

    def stage(name, fn, iters=100):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(dev)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize(dev)
        ms = e0.elapsed_time(e1) / iters
        print(f"{name:36s} {ms:8.2f} ms")
        return ms

    def fwd():
        return net(xpub, phi, w, seg, nseg)

    def fwd_policy():
        return losses(net, xpub, phi, w, seg, y, nseg, policy=policy, wp=0.05)

    def step():
        opt.zero_grad(set_to_none=True)
        fwd_policy().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

    for _ in range(3):
        buf.gather(ids)
    t0 = time.perf_counter()
    for _ in range(100):
        buf.gather(ids)
    print(f"{'gather (256 rows, wall)':36s} {1e3 * (time.perf_counter() - t0) / 100:8.2f} ms")
    stage("make_batch (mirror+expand)", lambda: make_gpu_batch(device_parts, np.random.default_rng(0), dev))
    cards = net.cards(xpub)
    physical = xpub[0::2]
    toks = net.tokens(physical, cards[0::2])
    stage("cards+tokens", lambda: (net.cards(xpub), net.tokens(physical, cards[0::2])))
    stage("trunk+board (256 boards)", lambda: net.board(physical, toks))
    stage("configs (12.3k)", lambda: net.configs(phi, cards[:, :5], seg))
    p = net.board(physical, toks)
    stage("heads+join", lambda: net.heads(p, net.configs(phi, cards[:, :5], seg)[1], w, seg, nseg))
    stage("net.forward (value only)", fwd)
    stage("forward + policy loss", fwd_policy)
    stage("backward + clip + adam", step)
    stage("full step (sync-free)", step)

    try:
        compiled = torch.compile(fwd_policy, mode="reduce-overhead", dynamic=True)
        stage("forward+policy (torch.compile)", compiled, iters=30)

        def compiled_step():
            opt.zero_grad(set_to_none=True)
            compiled().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        stage("full step (torch.compile)", compiled_step, iters=30)
    except Exception as e:
        print("compile unavailable:", type(e).__name__, str(e)[:120])
    try:
        g = torch.cuda.CUDAGraph()
        static = [torch.empty_like(x) for x in (xpub, phi, w, seg, y)]
        for s_, x_ in zip(static, (xpub, phi, w, seg, y)):
            s_.copy_(x_)
        with torch.cuda.graph(g):
            net(*static[:4], nseg)
        stage("forward (cuda graph)", lambda: g.replay(), iters=200)
    except Exception as e:
        print("cuda graph unavailable:", type(e).__name__, str(e)[:120])


if __name__ == "__main__":
    main()
