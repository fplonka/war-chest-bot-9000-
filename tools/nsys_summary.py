#!/usr/bin/env python3
"""Compact, comparable metrics from an Nsight Systems SQLite export.

Nsight's stock reports are intentionally exhaustive. This companion report
answers the questions used in the CUDA edit loop: how busy the device was,
where idle gaps remain, which kernels own time and launches, how much build
overlaps compute, and whether host API latency or device work is the ceiling.
"""

from __future__ import annotations

import argparse
import collections
import math
import sqlite3
from pathlib import Path


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    at = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return float(values[at])


def merged(intervals: list[tuple[int, int]]) -> tuple[int, list[int], int, int]:
    """Union duration, positive gaps, first timestamp, last timestamp."""
    if not intervals:
        return 0, [], 0, 0
    intervals.sort()
    first = intervals[0][0]
    lo, hi = intervals[0]
    total = 0
    gaps: list[int] = []
    for start, end in intervals[1:]:
        if start > hi:
            total += hi - lo
            gaps.append(start - hi)
            lo, hi = start, end
        else:
            hi = max(hi, end)
    total += hi - lo
    return total, gaps, first, hi


def fmt_ms(ns: float) -> str:
    return f"{ns / 1e6:10.3f} ms"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("--top", type=int, default=24)
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    kernels = [tuple(r) for r in db.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    )]
    copies = [tuple(r) for r in db.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
    )]
    sets = [tuple(r) for r in db.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start"
    )]
    gpu_ops = kernels + copies + sets
    busy, gaps, first, last = merged(gpu_ops)
    kernel_busy, _, _, _ = merged(kernels)
    span = last - first
    sum_kernel = sum(end - start for start, end in kernels)

    print("== GPU timeline ==")
    print(f"GPU span             {fmt_ms(span)}")
    print(f"any-op busy          {fmt_ms(busy)}  {100 * busy / span:6.2f}% of span")
    print(f"kernel busy          {fmt_ms(kernel_busy)}  {100 * kernel_busy / span:6.2f}% of span")
    print(f"summed kernel time   {fmt_ms(sum_kernel)}  {sum_kernel / kernel_busy:6.3f}x concurrency")
    print(f"kernel launches      {len(kernels):10d}")
    print(f"memcpy operations    {len(copies):10d}")
    print(f"memset operations    {len(sets):10d}")
    print(f"idle gaps            {len(gaps):10d}  total {fmt_ms(sum(gaps))}")
    print(
        "idle gap p50/p90/p99 "
        f"{percentile(gaps, .50) / 1e3:8.2f} / "
        f"{percentile(gaps, .90) / 1e3:8.2f} / "
        f"{percentile(gaps, .99) / 1e3:8.2f} us"
    )
    for threshold_us in (1, 5, 10, 50, 100):
        selected = [g for g in gaps if g >= threshold_us * 1000]
        print(
            f"gaps >= {threshold_us:3d} us    {len(selected):10d}  "
            f"total {fmt_ms(sum(selected))}"
        )

    print("\n== streams ==")
    stream_rows = db.execute(
        "SELECT streamId, COUNT(*) AS n, SUM(end-start) AS ns "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY streamId ORDER BY ns DESC"
    ).fetchall()
    for row in stream_rows:
        intervals = [tuple(r) for r in db.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId=? ORDER BY start",
            (row["streamId"],),
        )]
        union, _, sfirst, slast = merged(intervals)
        print(
            f"stream {row['streamId']:4d}  launches {row['n']:9d}  "
            f"sum {fmt_ms(row['ns'])}  union {fmt_ms(union)}  "
            f"active-span {(slast - sfirst) / 1e9:7.3f} s"
        )

    print("\n== kernels ==")
    rows = db.execute(
        "SELECT s.value AS name, COUNT(*) AS n, SUM(k.end-k.start) AS ns, "
        "AVG(k.end-k.start) AS avg_ns, AVG(k.gridX) AS grid_x, "
        "MAX(k.registersPerThread) AS regs, MAX(k.localMemoryPerThread) AS local_b "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName "
        "GROUP BY k.shortName ORDER BY ns DESC"
    ).fetchall()
    print(" time%   total-ms  launches    avg-us  avg-grid  regs localB  name")
    for row in rows[: args.top]:
        print(
            f"{100 * row['ns'] / sum_kernel:6.2f} "
            f"{row['ns'] / 1e6:10.3f} {row['n']:9d} "
            f"{row['avg_ns'] / 1e3:9.3f} {row['grid_x']:9.1f} "
            f"{row['regs']:5d} {row['local_b']:6d}  {row['name']}"
        )

    print("\n== kernel categories ==")
    categories: collections.Counter[str] = collections.Counter()
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        name = row["name"]
        if name == "backprop_level_flat":
            cat = "backward tree sweep"
        elif name in {"reach_level_flat", "reach_seed_flat", "reach_prop"}:
            cat = "forward reach sweep"
        elif name in {"regret_match_flat", "average_flat"}:
            cat = "regret + average"
        elif name in {"belief_sums_flat", "head_entry_flat", "readout_flat"}:
            cat = "belief/head/readout"
        elif "gemm" in name.lower() or "cutlass" in name.lower():
            cat = "GEMM"
        elif name in {
            "pack_cards", "cards_finish", "pile_pe", "pack_piles", "assemble",
            "trunk_norm", "holding_in", "slot_sum", "scatter_h0", "scatter_zg",
            "bias_act", "init_strategy", "seed_avg", "seed_snapshot_beliefs",
        }:
            cat = "admission/build"
        else:
            cat = "other"
        categories[cat] += row["ns"]
        counts[cat] += row["n"]
    for cat, ns in categories.most_common():
        print(f"{100 * ns / sum_kernel:6.2f}% {ns / 1e6:10.3f} ms {counts[cat]:9d} launches  {cat}")

    print("\n== memory operations ==")
    copy_kinds = dict(db.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER").fetchall())
    for row in db.execute(
        "SELECT copyKind,COUNT(*) AS n,SUM(bytes) AS bytes,SUM(end-start) AS ns "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind ORDER BY ns DESC"
    ):
        seconds = row["ns"] / 1e9
        gb = row["bytes"] / 1e9
        print(
            f"{copy_kinds.get(row['copyKind'], row['copyKind'])!s:18} "
            f"{row['n']:8d} ops {gb:9.3f} GB {row['ns'] / 1e6:9.3f} ms "
            f"{gb / seconds if seconds else math.nan:8.2f} GB/s"
        )
    row = db.execute(
        "SELECT COUNT(*) AS n,SUM(bytes) AS bytes,SUM(end-start) AS ns "
        "FROM CUPTI_ACTIVITY_KIND_MEMSET"
    ).fetchone()
    print(
        f"memset             {row['n']:8d} ops {row['bytes'] / 1e9:9.3f} GB "
        f"{row['ns'] / 1e6:9.3f} ms"
    )

    print("\n== host CUDA APIs ==")
    api_rows = db.execute(
        "SELECT s.value AS name,COUNT(*) AS n,SUM(r.end-r.start) AS ns,"
        "AVG(r.end-r.start) AS avg_ns FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId GROUP BY r.nameId ORDER BY ns DESC LIMIT 20"
    ).fetchall()
    total_api = sum(r["ns"] for r in api_rows)
    print(" total-ms   calls    avg-us  name")
    for row in api_rows:
        print(
            f"{row['ns'] / 1e6:9.3f} {row['n']:8d} "
            f"{row['avg_ns'] / 1e3:9.3f}  {row['name']}"
        )
    print(f"top-20 API time     {fmt_ms(total_api)}")


if __name__ == "__main__":
    main()
