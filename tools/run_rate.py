"""Compare training runs at equal workload depth, not equal wall time.

A ReBeL run's cost per solve grows as its games leave the opening, so a faster
build reaches the expensive positions sooner and its *average* rate over a
fixed window converges back towards a slower build's. Comparing the
instantaneous rate at the same cumulative solve count separates the two.

    python3 tools/run_rate.py runs/a/train.log runs/b/train.log
"""

import re
import sys

LINE = re.compile(r"\[t=\s*([0-9.]+)s\] rebel stream solves=(\d+) ")


def load(path):
    rows = []
    for line in open(path):
        m = LINE.search(line)
        if m:
            rows.append((float(m.group(1)), int(m.group(2))))
    return rows


def rate_at(rows, n, window=100_000):
    """Solves per second over a window of solves centred on `n`.

    A single ten-second interval is far too noisy to compare: one snapshot
    pause or one very large wave moves it by a third.
    """
    lo, hi = n - window, n + window
    a = next((r for r in rows if r[1] >= lo), None)
    b = next((r for r in reversed(rows) if r[1] <= hi), None)
    if a is None or b is None or b[0] <= a[0] or b[1] <= a[1]:
        return None
    return (b[1] - a[1]) / (b[0] - a[0])


def main():
    runs = [(p, load(p)) for p in sys.argv[1:]]
    names = [p.split("/")[-2] for p, _ in runs]
    print("%10s " % "solves" + " ".join(f"{n:>16}" for n in names))
    for n in range(200_000, 1_600_001, 200_000):
        cells = []
        for _, rows in runs:
            r = rate_at(rows, n)
            cells.append(f"{r:>16.0f}" if r else f"{'-':>16}")
        print(f"{n:>10} " + " ".join(cells))
    for name, (_, rows) in zip(names, runs):
        print(f"{name}: {rows[-1][1]} solves at t={rows[-1][0]:.0f}s")


if __name__ == "__main__":
    main()
