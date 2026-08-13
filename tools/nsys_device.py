"""Per-device occupancy from an Nsight Systems SQLite export.

`tools/nsys_summary.py` merges both cards into one timeline, which hides the
question that matters here: is each GPU actually running kernels, and how much
of the SM capacity does the resident work ask for. A card that is "busy" with
one 3-block GEMM is idle in every sense that matters.

    python3 tools/nsys_device.py capture.sqlite
"""

import collections
import sqlite3
import sys


def union(intervals):
    intervals.sort()
    total = 0
    lo, hi = intervals[0]
    for start, end in intervals[1:]:
        if start > hi:
            total += hi - lo
            lo, hi = start, end
        else:
            hi = max(hi, end)
    return total + hi - lo


def main():
    db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = list(db.execute("""
        SELECT deviceId, start, end, gridX * gridY * gridZ AS blocks,
               blockX * blockY * blockZ AS threads
        FROM CUPTI_ACTIVITY_KIND_KERNEL
    """))
    per = collections.defaultdict(list)
    for r in rows:
        per[r["deviceId"]].append(r)
    span_lo = min(r["start"] for r in rows)
    span_hi = max(r["end"] for r in rows)
    span = span_hi - span_lo
    print(f"span {span / 1e6:.1f} ms   kernels {len(rows)}")
    for device in sorted(per):
        k = per[device]
        busy = union([(r["start"], r["end"]) for r in k])
        summed = sum(r["end"] - r["start"] for r in k)
        # Thread-seconds asked for, against what the card could have run in the
        # same wall time. A 3090 holds 82 SMs x 1536 resident threads.
        capacity = 82 * 1536
        demand = sum((r["end"] - r["start"]) * min(r["blocks"] * r["threads"],
                                                   capacity) for r in k)
        print(f"  device {device}: kernels {len(k):7d}"
              f"  busy {100 * busy / span:5.1f}%"
              f"  summed/span {summed / span:5.2f}x"
              f"  mean resident threads {demand / span / capacity * 100:5.1f}%"
              f"  median blocks {sorted(r['blocks'] for r in k)[len(k) // 2]:6d}")




def concurrency(path):
    """How many of a card's lanes have a kernel running at once.

    If the answer is mostly 0 or 1 while five lanes are queued, the lanes are
    convoyed and the card's idle time is recoverable by scheduling. If it is
    mostly 1 with short gaps, the card is genuinely serialised by dependent
    work and only less work per solve will help.
    """
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = list(db.execute("""
        SELECT deviceId, streamId, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
    """))
    per = collections.defaultdict(list)
    for r in rows:
        per[r["deviceId"]].append(r)
    for device in sorted(per):
        events = []
        streams = collections.Counter()
        for r in per[device]:
            events.append((r["start"], 1))
            events.append((r["end"], -1))
            streams[r["streamId"]] += r["end"] - r["start"]
        events.sort()
        held = collections.Counter()
        level = 0
        last = events[0][0]
        for at, delta in events:
            held[level] += at - last
            last = at
            level += delta
        span = events[-1][0] - events[0][0]
        dist = " ".join(f"{k}:{100 * v / span:.1f}%"
                        for k, v in sorted(held.items()) if v / span > 0.002)
        print(f"  device {device} concurrent kernels  {dist}")
        busy = " ".join(f"{100 * v / span:.0f}%" for _, v in
                        sorted(streams.items(), key=lambda x: -x[1]))
        print(f"    per-stream busy: {busy}")


if __name__ == "__main__":
    main()
    concurrency(sys.argv[1])
