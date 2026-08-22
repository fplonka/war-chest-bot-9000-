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


def category(name: str) -> str:
    if name == "backprop_level_flat":
        return "backward tree sweep"
    if name in {"reach_level_flat", "reach_seed_flat", "reach_prop"}:
        return "forward reach sweep"
    if name in {"regret_match_flat", "average_flat"}:
        return "regret + average"
    if name in {"belief_sums_flat", "head_entry_flat", "readout_flat"}:
        return "belief/head/readout"
    if "gemm" in name.lower() or "cutlass" in name.lower():
        return "GEMM"
    if name in {
        "pack_cards", "cards_finish", "pile_pe", "pack_piles", "assemble",
        "trunk_norm", "holding_in", "slot_sum", "scatter_h0", "scatter_zg",
        "bias_act", "init_strategy", "seed_avg", "seed_snapshot_beliefs",
    }:
        return "admission/build"
    return "other"


def where(device: int | None) -> str:
    return "" if device is None else " WHERE deviceId=?"


def arg(device: int | None) -> tuple:
    return () if device is None else (device,)


def gap_hist(gaps: list[int]) -> None:
    print(
        "idle gap p50/p90/p99 "
        f"{percentile(gaps, .50) / 1e3:8.2f} / "
        f"{percentile(gaps, .90) / 1e3:8.2f} / "
        f"{percentile(gaps, .99) / 1e3:8.2f} us"
    )
    edges = [0, 1, 5, 10, 50, 100, 500, 1000, 5000]
    for lo, hi in zip(edges, edges[1:] + [None]):
        lo_ns = lo * 1000
        selected = (
            [g for g in gaps if g >= lo_ns]
            if hi is None
            else [g for g in gaps if lo_ns <= g < hi * 1000]
        )
        label = f">= {lo:4d} us" if hi is None else f"{lo:4d}-{hi:<4d} us"
        print(
            f"gaps {label:14} {len(selected):10d}  "
            f"total {fmt_ms(sum(selected))}"
        )


def report_device(db: sqlite3.Connection, device: int | None, top: int) -> None:
    tag = "merged across cards" if device is None else f"device {device}"
    w, a = where(device), arg(device)
    kernels = [tuple(r) for r in db.execute(
        f"SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL{w} ORDER BY start", a
    )]
    copies = [tuple(r) for r in db.execute(
        f"SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY{w} ORDER BY start", a
    )]
    sets = [tuple(r) for r in db.execute(
        f"SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET{w} ORDER BY start", a
    )]
    gpu_ops = kernels + copies + sets
    busy, gaps, first, last = merged(gpu_ops)
    kernel_busy, _, _, _ = merged(kernels)
    span = last - first
    sum_kernel = sum(end - start for start, end in kernels)
    print(f"\n== GPU timeline ({tag}) ==")
    if span <= 0:
        print("no GPU ops")
        return
    print(f"GPU span             {fmt_ms(span)}")
    print(f"any-op busy          {fmt_ms(busy)}  {100 * busy / span:6.2f}% of span")
    print(f"kernel busy          {fmt_ms(kernel_busy)}  {100 * kernel_busy / span:6.2f}% of span")
    conc = sum_kernel / kernel_busy if kernel_busy else 0.0
    print(f"summed kernel time   {fmt_ms(sum_kernel)}  {conc:6.3f}x concurrency")
    print(f"kernel launches      {len(kernels):10d}")
    print(f"memcpy operations    {len(copies):10d}")
    print(f"memset operations    {len(sets):10d}")
    print(f"idle gaps            {len(gaps):10d}  total {fmt_ms(sum(gaps))}")
    gap_hist(gaps)

    print(f"\n== streams ({tag}) ==")
    stream_rows = db.execute(
        f"SELECT streamId, COUNT(*) AS n, SUM(end-start) AS ns "
        f"FROM CUPTI_ACTIVITY_KIND_KERNEL{w} GROUP BY streamId ORDER BY ns DESC",
        a,
    ).fetchall()
    for row in stream_rows:
        extra = a + (row["streamId"],)
        filt = "deviceId=? AND " if device is not None else ""
        intervals = [tuple(r) for r in db.execute(
            f"SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL "
            f"WHERE {filt}streamId=? ORDER BY start",
            extra,
        )]
        union, _, sfirst, slast = merged(intervals)
        print(
            f"stream {row['streamId']:4d}  launches {row['n']:9d}  "
            f"sum {fmt_ms(row['ns'])}  union {fmt_ms(union)}  "
            f"active-span {(slast - sfirst) / 1e9:7.3f} s"
        )

    print(f"\n== kernels ({tag}) ==")
    rows = db.execute(
        f"SELECT s.value AS name, COUNT(*) AS n, SUM(k.end-k.start) AS ns, "
        f"AVG(k.end-k.start) AS avg_ns, AVG(k.gridX) AS grid_x, "
        f"MAX(k.registersPerThread) AS regs, MAX(k.localMemoryPerThread) AS local_b "
        f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName"
        f"{w.replace('deviceId', 'k.deviceId')} "
        f"GROUP BY k.shortName ORDER BY ns DESC",
        a,
    ).fetchall()
    print(" time%   total-ms  launches    avg-us  avg-grid  regs localB  name")
    denom = sum_kernel or 1
    for row in rows[:top]:
        print(
            f"{100 * row['ns'] / denom:6.2f} "
            f"{row['ns'] / 1e6:10.3f} {row['n']:9d} "
            f"{row['avg_ns'] / 1e3:9.3f} {row['grid_x']:9.1f} "
            f"{row['regs']:5d} {row['local_b']:6d}  {row['name']}"
        )

    print(f"\n== kernel categories ({tag}) ==")
    categories: collections.Counter[str] = collections.Counter()
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        cat = category(row["name"])
        categories[cat] += row["ns"]
        counts[cat] += row["n"]
    for cat, ns in categories.most_common():
        print(f"{100 * ns / denom:6.2f}% {ns / 1e6:10.3f} ms {counts[cat]:9d} launches  {cat}")

    print(f"\n== memory operations ({tag}) ==")
    copy_kinds = dict(db.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER").fetchall())
    for row in db.execute(
        f"SELECT copyKind,COUNT(*) AS n,SUM(bytes) AS bytes,SUM(end-start) AS ns "
        f"FROM CUPTI_ACTIVITY_KIND_MEMCPY{w} GROUP BY copyKind ORDER BY ns DESC",
        a,
    ):
        seconds = row["ns"] / 1e9
        gb = (row["bytes"] or 0) / 1e9
        print(
            f"{copy_kinds.get(row['copyKind'], row['copyKind'])!s:18} "
            f"{row['n']:8d} ops {gb:9.3f} GB {row['ns'] / 1e6:9.3f} ms "
            f"{gb / seconds if seconds else math.nan:8.2f} GB/s"
        )
    row = db.execute(
        f"SELECT COUNT(*) AS n,SUM(bytes) AS bytes,SUM(end-start) AS ns "
        f"FROM CUPTI_ACTIVITY_KIND_MEMSET{w}",
        a,
    ).fetchone()
    print(
        f"memset             {row['n']:8d} ops {(row['bytes'] or 0) / 1e9:9.3f} GB "
        f"{(row['ns'] or 0) / 1e6:9.3f} ms"
    )


def report_host_apis(db: sqlite3.Connection) -> None:
    print("\n== host CUDA APIs ==")
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
        print("no CUPTI runtime table")
        return
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

    cols = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME)")}
    tid_col = "globalTid" if "globalTid" in cols else ("threadId" if "threadId" in cols else None)
    if not tid_col:
        print("no thread id on runtime table")
        return
    print(f"\n== host CUDA APIs by thread ({tid_col}) ==")
    thread_tot = db.execute(
        f"SELECT {tid_col} AS tid, COUNT(*) AS n, SUM(end-start) AS ns "
        f"FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY {tid_col} ORDER BY ns DESC LIMIT 12"
    ).fetchall()
    for th in thread_tot:
        print(
            f"\nthread {th['tid']}  calls {th['n']:8d}  "
            f"sum {fmt_ms(th['ns'])}"
        )
        for row in db.execute(
            f"SELECT s.value AS name,COUNT(*) AS n,SUM(r.end-r.start) AS ns,"
            f"AVG(r.end-r.start) AS avg_ns FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            f"JOIN StringIds s ON s.id=r.nameId WHERE r.{tid_col}=? "
            f"GROUP BY r.nameId ORDER BY ns DESC LIMIT 8",
            (th["tid"],),
        ):
            print(
                f"  {row['ns'] / 1e6:9.3f} {row['n']:8d} "
                f"{row['avg_ns'] / 1e3:9.3f}  {row['name']}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("--top", type=int, default=24)
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        print("no CUPTI kernel table: empty trace or CUPTI denied")
        print("tables:", ", ".join(sorted(tables)[:40]))
        return

    devices = [r[0] for r in db.execute(
        "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId"
    )]
    print(f"devices with kernels: {devices}")
    # Merged-across-cards occupancy is not occupancy: two half-busy cards look
    # full. Keep it as a launch census, then print each card on its own timeline.
    report_device(db, None, args.top)
    for dev in devices:
        report_device(db, dev, args.top)
    report_host_apis(db)


if __name__ == "__main__":
    main()
