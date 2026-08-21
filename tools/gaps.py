"""What the cards wait for.

Kernel-busy is two thirds of the span, and the missing third is in gaps. A gap
belongs to whatever ran before it: if one kernel ends most of the long ones,
the round is waiting on that kernel's result rather than on the cards.
"""
import sqlite3, sys, collections

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
rows = db.execute("""
    SELECT k.start, k.end, k.deviceId, s.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName
    ORDER BY k.deviceId, k.start
""").fetchall()

# Per device, the wall the card is idle after each kernel: the next start on
# that card, minus this end. Overlapping kernels give a negative gap and count
# as none.
after = collections.Counter()
count = collections.Counter()
by_dev = collections.defaultdict(list)
for st, en, dev, name in rows:
    by_dev[dev].append((st, en, name.split("(")[0]))
for ks in by_dev.values():
    end, prev = None, None
    for st, en, name in ks:
        if end is not None and prev is not None and st > end:
            after[prev] += st - end
            count[prev] += 1
        end = en if end is None else max(end, en)
        prev = name
total = sum(after.values())
print(f"idle after a kernel: {total / 1e9:.2f}s")
for name, ns in after.most_common(10):
    print(f"  {ns / 1e6:9.1f} ms  {100 * ns / total:5.1f}%  {count[name]:7d} gaps  "
          f"{ns / max(count[name], 1) / 1e3:8.1f} us each  {name}")
