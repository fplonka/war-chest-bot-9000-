"""CPU-only round-trip coverage for the replay snapshot state."""

import importlib.util
import pathlib
import sys
import types

import numpy as np


def load_buffer_module():
    """Load train.py with only the constants needed by Buffer."""
    warchest = types.ModuleType("warchest")
    for name, value in {
        "PUBFEAT": 1,
        "CFEAT": 1,
        "CCOUNTS": 5,
        "CNORM": 1.0,
        "ROW_BYTES": 7,
        "ACT_BYTES": 4,
        "N_KINDS": 1,
        "NSLOT": 1,
        "N_HEXES": 3,
    }.items():
        setattr(warchest, name, value)
    torch = types.ModuleType("torch")
    torch.no_grad = lambda: (lambda fn: fn)
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    value_net = types.ModuleType("value_net")
    value_net.AFEAT = 1
    value_net.Net = object
    config = types.ModuleType("config")
    mirror = types.ModuleType("mirror")
    export_weights = types.ModuleType("export_weights")
    export_weights.load = lambda _: None
    sys.modules.update({
        "warchest": warchest,
        "torch": torch,
        "torch.nn": torch.nn,
        "torch.nn.functional": torch.nn.functional,
        "value_net": value_net,
        "config": config,
        "mirror": mirror,
        "export_weights": export_weights,
    })
    path = pathlib.Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("train_buffer_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def chunk(train, start, n):
    ids = np.arange(start, start + n)
    p0 = 1 + ids % 4
    p1 = 2 + ids % 3
    lens = np.column_stack((p0, p1))
    coff = np.concatenate(([0], np.cumsum(lens.ravel()))).astype(np.int64)
    m = int(coff[-1])
    rows = (np.arange(n * train.ROW_BYTES, dtype=np.uint32) + start).astype(
        np.uint8).reshape(n, train.ROW_BYTES)
    cc = np.arange(m * train.CCOUNTS, dtype=np.uint32).astype(np.uint8).reshape(
        m, train.CCOUNTS)
    cw = np.concatenate([
        np.full(k, 1.0 / k, np.float32)
        for row in lens for k in row
    ])
    cy = np.linspace(-1.0, 1.0, m, dtype=np.float32)
    boundaries = np.arange(0, n + 1, 97, dtype=np.int64)
    if boundaries[-1] != n:
        boundaries = np.append(boundaries, n)
    source = (ids % 3).astype(np.uint8)
    truth = np.zeros((n, 2), np.uint32)
    outcome = np.full((n, 2), np.nan, np.float32)
    outcome[ids % 5 == 0] = (0.25, -0.25)
    created = (1000.0 + ids).astype(np.float64)
    td1 = (ids % 7 == 0).astype(np.uint8)

    has_policy = ids % 3 != 0
    na_row = np.where(has_policy, 1 + ids % 3, 0)
    nc_row = np.where(has_policy, 1 + ids % 4, 0)
    paoff = np.concatenate(([0], np.cumsum(na_row))).astype(np.int64)
    pcoff = np.concatenate(([0], np.cumsum(nc_row))).astype(np.int64)
    pa = np.arange(int(paoff[-1]) * train.ACT_BYTES, dtype=np.uint32).astype(
        np.uint8).reshape(-1, train.ACT_BYTES)
    pci = np.concatenate([
        (np.arange(nc) % int(p0[i] + p1[i]))
        for i, nc in enumerate(nc_row) if nc
    ]).astype(np.uint16)
    pact = np.concatenate([
        np.arange(nc) % int(na_row[i])
        for i, nc in enumerate(nc_row) if nc
    ]).astype(np.uint16)
    pprob = np.concatenate([
        np.full(nc, 1.0 / nc, np.float32)
        for nc in nc_row if nc
    ])
    policy = (pa, paoff, pcoff, pci, pact, pprob)
    return (rows, cc, cw, cy, coff, boundaries, source, truth, outcome,
            created, td1, policy)


def assert_same(left, right):
    if isinstance(left, tuple):
        assert isinstance(right, tuple) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
        return
    np.testing.assert_array_equal(left, right)


def main():
    train = load_buffer_module()
    train.time.time = lambda: 1_000_000.0
    buf = train.Buffer(4096, 12_000)
    for start in range(0, 6000, 1000):
        buf.add(*chunk(train, start, 1000))
    state = buf.state_dict()
    assert len(state["x"]) == len(buf) < buf.cap
    assert len(state["cc"]) < buf.ccap
    assert len(state["pa"]) < buf.acap
    assert len(state["pci"]) < buf.pcap

    restored = train.Buffer(buf.cap, buf.ccap)
    restored.add(*chunk(train, 7000, 1000))
    restored.load_state_dict(state)
    assert len(restored) == len(buf)
    ids = np.arange(buf.lo, buf.rows)
    assert_same(buf.gather(ids), restored.gather(ids))
    assert_same(buf.ordered(), restored.ordered())
    assert buf.replay_stats() == restored.replay_stats()
    assert buf.span_seconds() == restored.span_seconds()
    np.testing.assert_array_equal(buf.soff, restored.soff)

    left = buf.sample_calibration(128, np.random.default_rng(91))
    right = restored.sample_calibration(128, np.random.default_rng(91))
    assert_same(left, right)
    for kwargs in ({}, {"recent_mix": 0.37, "recent_frac": 0.31}):
        left = buf.sample_ids(500, np.random.default_rng(73), **kwargs)
        right = restored.sample_ids(500, np.random.default_rng(73), **kwargs)
        np.testing.assert_array_equal(left, right)

    extra = chunk(train, 6000, 1000)
    buf.add(*extra)
    restored.add(*extra)
    ids = np.arange(buf.lo, buf.rows)
    assert_same(buf.gather(ids), restored.gather(ids))
    np.testing.assert_array_equal(buf.soff, restored.soff)

    bad = {
        "rows": np.zeros((1, train.ROW_BYTES), np.uint8),
        "cc": np.zeros((1, train.CCOUNTS), np.uint8),
        "cw": np.array([np.nan, np.inf], np.float32),
        "cy": np.array([np.nan], np.float32),
    }
    try:
        train.ingest(train.Buffer(4, 16), bad)
    except SystemExit as exc:
        message = str(exc)
        assert "data['cw']=2" in message and "data['cy']=1" in message
    else:
        raise AssertionError("non-finite collect values were accepted")
    print("buffer snapshot test OK")


if __name__ == "__main__":
    main()
