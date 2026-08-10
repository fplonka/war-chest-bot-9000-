"""Summarise a WARCHEST_GPU_PROFILE stream log.

Answers one question: how much of the wall clock is each GPU lane actually
inside a wave, and where does that time go. A lane that is busy far less than
the wall clock is starved, not slow, and the fix is upstream of CUDA.

    python3 tools/wave_profile.py runs/<tag>/stream.log
"""

import collections
import re
import statistics
import sys

FIELDS = re.compile(r"([a-z_]+)=([0-9.]+)")


def parse(path):
    service = collections.defaultdict(list)
    device = []
    wall = None
    for line in open(path):
        if line.startswith("v5_service"):
            row = {k: float(v) for k, v in FIELDS.findall(line)}
            service[(int(row["device"]), int(row["lane"]))].append(row)
        elif line.startswith("v5_device"):
            device.append({k: float(v) for k, v in FIELDS.findall(line)})
        elif '"phase": "prestop"' in line:
            wall = float(re.search(r'"seconds": ([0-9.]+)', line).group(1))
    return service, device, wall


def main():
    service, device, wall = parse(sys.argv[1])
    wall = wall or float(sys.argv[2])
    print(f"wall {wall:.1f}s  lanes {len(service)}  "
          f"service waves {sum(len(v) for v in service.values())}  "
          f"device waves {len(device)}")

    busy = 0.0
    for key in sorted(service):
        rows = service[key]
        solve = sum(x["solve_ms"] for x in rows) / 1e3
        pack = sum(x["pack_ms"] for x in rows) / 1e3
        busy += solve + pack
        classes = collections.Counter(int(x["class"]) for x in rows)
        print(
            f"  dev{key[0]} lane{key[1]}: waves {len(rows):5d}"
            f"  solve {solve:6.1f}s ({100 * solve / wall:5.1f}%)"
            f"  pack {pack:5.1f}s"
            f"  jobs~{statistics.median(x['jobs'] for x in rows):4.0f}"
            f"  rows~{statistics.median(x['rows'] for x in rows):7.0f}"
            f"  cells~{statistics.median(x['cells'] for x in rows):8.0f}"
            f"  wait~{statistics.median(x['oldest_ms'] for x in rows):6.1f}ms"
            f"  classes {dict(sorted(classes.items()))}"
        )
    lanes = max(len(service), 1)
    print(f"  lane occupancy {100 * busy / (wall * lanes):.1f}% "
          f"({busy:.0f}s busy of {wall * lanes:.0f}s available)")

    if device:
        print("  device phase totals (all lanes):")
        for k in ("upload_ms", "capture_ms", "queue_ms", "gpu_ms", "unpack_ms",
                  "total_ms"):
            values = [x[k] for x in device]
            print(f"    {k:10} sum {sum(values) / 1e3:7.1f}s"
                  f"  mean {statistics.fmean(values):7.2f}ms"
                  f"  p95 {sorted(values)[int(0.95 * len(values))]:8.2f}ms")
        rows = [x["rows"] for x in device]
        jobs = [x["jobs"] for x in device]
        print(f"    rows  median {statistics.median(rows):8.0f}"
              f"  mean {statistics.fmean(rows):8.0f}"
              f"  p95 {sorted(rows)[int(0.95 * len(rows))]:8.0f}")
        print(f"    jobs  median {statistics.median(jobs):8.0f}"
              f"  mean {statistics.fmean(jobs):8.1f}"
              f"  p95 {sorted(jobs)[int(0.95 * len(jobs))]:8.0f}")




def buckets(path):
    """Per-wave cost against wave size: is a bigger wave cheaper per solve?"""
    _, device, _ = parse(path)
    groups = {}
    for row in device:
        j = int(row["jobs"])
        key = 1 << max(0, (j - 1).bit_length())
        groups.setdefault(key, []).append(row)
    head = ("bucket", "n", "jobs", "rows", "cells", "upload", "capture",
            "queue", "unpack", "total", "ms/solve")
    print("%7s %5s %7s %9s %11s %8s %8s %9s %7s %9s %9s" % head)
    for key in sorted(groups):
        v = groups[key]
        f = lambda k: statistics.fmean(x[k] for x in v)
        print("%7d %5d %7.1f %9.0f %11.0f %8.2f %8.2f %9.2f %7.2f %9.2f %9.2f" % (
            key, len(v), f("jobs"), f("rows"), f("cells"), f("upload_ms"),
            f("capture_ms"), f("queue_ms"), f("unpack_ms"), f("total_ms"),
            f("total_ms") / max(f("jobs"), 1e-9)))


if __name__ == "__main__":
    main()
    print()
    buckets(sys.argv[1])
