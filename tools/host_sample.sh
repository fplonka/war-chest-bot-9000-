#!/usr/bin/env bash
# Sample host threads of a live train.py after a delay. Writes under $1.
set -euo pipefail
out=${1:?out dir}
delay=${2:-105}
mkdir -p "$out"
sleep "$delay"
pid=$(pgrep -n -f 'python(3)? train/train.py' || true)
date -u +%FT%TZ > "$out/host_sample_time.txt"
echo "pid=$pid" >> "$out/host_sample_time.txt"
if [ -z "$pid" ]; then
    echo "no train.py" >> "$out/host_sample_time.txt"
    exit 0
fi
top -H -b -n 3 -d 1 -p "$pid" > "$out/topH.txt"
ps -eLo pid,tid,pcpu,stat,comm | awk -v p="$pid" 'NR==1 || $1==p' \
    | sort -k3 -nr > "$out/ps_threads.txt"
nvidia-smi > "$out/nvidia-smi.txt"
if command -v py-spy >/dev/null 2>&1; then
    py-spy dump --pid "$pid" --nonblocking > "$out/pyspy_dump.txt" 2>&1 || \
        py-spy dump --pid "$pid" > "$out/pyspy_dump.txt" 2>&1 || true
fi
# One-line CPU sums by thread name prefix.
python3 - "$out/ps_threads.txt" "$out/thread_cpu.txt" <<'PY'
import sys
from collections import defaultdict
src, dst = sys.argv[1], sys.argv[2]
by = defaultdict(float)
n = defaultdict(int)
with open(src) as f:
    next(f)
    for line in f:
        bits = line.split()
        if len(bits) < 5:
            continue
        cpu, name = float(bits[2]), bits[4]
        if name.startswith("host-"):
            key = "host-*"
        elif name.startswith("card-"):
            key = "card-*"
        else:
            key = name
        by[key] += cpu
        n[key] += 1
with open(dst, "w") as f:
    f.write(f"{'group':20} {'n':>4} {'sum%':>8}\n")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        f.write(f"{k:20} {n[k]:4d} {v:8.1f}\n")
PY
