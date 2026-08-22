#!/usr/bin/env bash
# The measurements behind docs/REDESIGN.md.
#
#   tools/probe.sh rate     solves/s against the number of host workers, and
#                           against the search budget
#   tools/probe.sh device   nsys: where the card's time goes, and what it waits
#                           for. Few workers, not many -- a full host under
#                           nsys never reaches a kernel launch
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
PROF=${PROF:-45}

bench() {  # bench <workers> <s> <c> [env...]
    local t=$1 s=$2 c=$3; shift 3
    echo "=== workers=$t s=$s c=$c $* ===" >> "$OUT"
    env "$@" setsid nohup python tools/farmbench.py \
        --roots "$ROOTS" --weights "$W" --devices 0,1 \
        --threads "$t" --s "$s" --c "$c" \
        --seconds "$SECS" --window 50 >> "$OUT" 2>&1 &
    wait $!
}

case "${1:-}" in
rate)
    OUT=/workspace/logs/rate.log; : > "$OUT"
    for t in 18 36 54 72; do bench "$t" 512 8; done        # host workers
    for i in 128 256 512; do bench 72 "$i" 1; done         # the budget, at c=1
    bench 72 512 4
    bench 72 512 8 WARCHEST_STAGES=1                       # where a round goes
    ;;
device)
    OUT=/workspace/logs/device.log; : > "$OUT"
    REP=/workspace/prof
    # `--duration` is what makes the report whole: nsys stops the target itself
    # when the clock runs out, so farmbench's intermittent shutdown deadlock
    # never gets a chance to invite a `kill -9`. A SIGKILL of the target keeps
    # only the periodic flushes, and a SIGKILL of nsys writes no report at all.
    # The target is given longer than the capture so the capture ends first.
    nsys profile -o "$REP" --force-overwrite true --duration="$PROF" \
        --trace=cuda --sample=none --cpuctxsw=none \
        python tools/farmbench.py --roots "$ROOTS" --weights "$W" \
        --devices 0,1 --threads 8 --seconds $((PROF + 30)) --window 40 \
        >> "$OUT" 2>&1
    nsys export --type sqlite -o "$REP.sqlite" --force-overwrite true \
        "$REP.nsys-rep" >> "$OUT" 2>&1
    python tools/nsys_summary.py "$REP.sqlite" >> "$OUT" 2>&1 || true
    python tools/gaps.py "$REP.sqlite" >> "$OUT" 2>&1 || true
    ;;
host)
    OUT=/workspace/logs/host.log; : > "$OUT"
    (cd engine && maturin develop --release --features python,gpu,prof) >> "$OUT" 2>&1
    bench 72 512 8          # every core advancing solves
    (cd engine && maturin develop --release --features python,gpu) >> "$OUT" 2>&1
    ;;
*)  sed -n '2,13p' "$0" ;;
esac
echo "done" >> "${OUT:-/dev/null}"
