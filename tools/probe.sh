#!/usr/bin/env bash
# The measurements behind docs/REDESIGN.md.
#
#   tools/probe.sh rate     solves/s against solves in flight, and against the
#                           search budget; also what a solver thread's
#                           turnaround is when it is not queueing for a core
#   tools/probe.sh device   nsys: where the card's time goes, and what it waits
#                           for. Two cohorts, not ten -- ten under nsys never
#                           reach a kernel launch
#   tools/probe.sh host     the per-phase host breakdown, under a `prof` build
#
# Nothing here may run beside anything else: a rate measured while the host is
# busy solving is a measurement of the contention.
set -uo pipefail
cd /workspace/warchest-engine
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
source /venv/main/bin/activate 2>/dev/null || true
mkdir -p /workspace/logs

ROOTS=${ROOTS:-/workspace/roots.bin}
W=${W:-runs/cohorts10/snap_02.pt}
SECS=${SECS:-150}

bench() {  # bench <cohorts> <threads> <s> <c> [env...]
    local co=$1 t=$2 s=$3 c=$4; shift 4
    echo "=== cohorts=$co threads=$t s=$s c=$c $* ===" >> "$OUT"
    env "$@" setsid nohup python tools/farmbench.py \
        --roots "$ROOTS" --weights "$W" --devices 0,1 \
        --threads "$t" --cohorts "$co" --s "$s" --c "$c" \
        --seconds "$SECS" --window 50 >> "$OUT" 2>&1 &
    wait $!
}

case "${1:-}" in
rate)
    OUT=/workspace/logs/rate.log; : > "$OUT"
    for c in 2 4 6 8 10 12; do bench "$c" 36 512 8; done   # solves in flight
    bench 10 2 512 8                                       # uncontended `awake`
    for i in 128 256 512; do bench 10 36 "$i" 1; done      # the budget, at c=1
    bench 10 36 512 4
    bench 10 36 512 8 WARCHEST_STAGES=1                     # where a round goes
    ;;
device)
    OUT=/workspace/logs/device.log; : > "$OUT"
    REP=/workspace/prof
    setsid nohup nsys profile -o "$REP" --force-overwrite true \
        --trace=cuda --sample=none --cpuctxsw=none \
        python tools/farmbench.py --roots "$ROOTS" --weights "$W" \
        --devices 0,1 --threads 36 --cohorts 2 --seconds 45 --window 40 \
        >> "$OUT" 2>&1 &
    wait $!
    nsys export --type sqlite -o "$REP.sqlite" --force-overwrite true \
        "$REP.nsys-rep" >> "$OUT" 2>&1
    python tools/nsys_summary.py "$REP.sqlite" >> "$OUT" 2>&1 || true
    python tools/gaps.py "$REP.sqlite" >> "$OUT" 2>&1 || true
    ;;
host)
    OUT=/workspace/logs/host.log; : > "$OUT"
    (cd engine && maturin develop --release --features python,gpu,prof) >> "$OUT" 2>&1
    bench 10 36 512 8       # 360 threads on 72: turnaround plus queueing
    bench 10 2 512 8        # 20 threads: turnaround alone
    (cd engine && maturin develop --release --features python,gpu) >> "$OUT" 2>&1
    ;;
*)  sed -n '2,13p' "$0" ;;
esac
echo "done" >> "${OUT:-/dev/null}"
