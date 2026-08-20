"""Does a cohort march in step?

A solve is a fixed number of rounds and every round advances every solve by
one, so threads that start together can stay together -- and then the device
sees all its trees tiny at once and all of them full at once. The width of a
per-level sweep is the tree size, so its grid over time is the answer.
"""
import sqlite3, sys, statistics

db = sqlite3.connect(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else "k_backprop_sweep"
rows = db.execute("""
    SELECT k.start, k.gridX FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON s.id = k.demangledName
    WHERE s.value LIKE ? ORDER BY k.start
""", (f"%{name}%",)).fetchall()
if not rows:
    print(f"no {name} launches"); sys.exit()
t0 = rows[0][0]
span = (rows[-1][0] - t0) / 1e9
print(f"{name}: {len(rows)} launches over {span:.1f}s  "
      f"grid mean {statistics.mean(r[1] for r in rows):.0f} "
      f"sd {statistics.pstdev(r[1] for r in rows):.0f}")
# Mean grid in each tenth of a second: a march shows as a swing, a spread as a line.
buckets = {}
for st, g in rows:
    buckets.setdefault(int((st - t0) / 2e8), []).append(g)
line = [statistics.mean(v) for _, v in sorted(buckets.items())]
lo, hi = min(line), max(line)
print(f"  per 0.2s: min {lo:.0f} max {hi:.0f} swing {hi / max(lo, 1):.1f}x")
print("  " + "".join(" .:-=+*#@"[min(8, int(8 * (v - lo) / max(hi - lo, 1e-9)))] for v in line[:110]))
