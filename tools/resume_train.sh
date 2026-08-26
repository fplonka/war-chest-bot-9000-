#!/usr/bin/env bash
# Run a trainer and resume its newest checkpoint after an unexpected exit.
set -euo pipefail

args=("$@")
out=
minutes=
for arg in "${args[@]}"; do
    case "$arg" in
        out=*) out=${arg#out=} ;;
        minutes=*) minutes=${arg#minutes=} ;;
    esac
done
[ -n "$out" ] || { echo "resume_train.sh needs out=<name>" >&2; exit 2; }
run=${out#runs/}
dir="runs/$run"

newest_snapshot() {
    ls -1t "$dir"/snap_*.pt 2>/dev/null | head -1
}

resume=$(newest_snapshot || true)
failures=0
while :; do
    if [ -n "$resume" ]; then
        command=(python train/train.py "out=$out" "resume=$resume")
        [ -n "$minutes" ] && command+=("minutes=$minutes")
    else
        command=(python train/train.py "${args[@]}")
    fi
    if "${command[@]}"; then
        exit 0
    else
        status=$?
    fi
    next=$(newest_snapshot || true)
    [ -n "$next" ] || exit "$status"
    if [ "$next" = "$resume" ]; then
        failures=$((failures + 1))
        if [ "$failures" -ge 3 ]; then
            echo "[box] trainer failed $failures times without a new snapshot; stopping" >&2
            exit "$status"
        fi
    else
        failures=0
    fi
    resume=$next
    echo "[box] trainer exited $status; resuming $resume" >&2
    sleep 5
done
