#!/usr/bin/env bash
# One mature-workload generation trace on the GPU box, with CPU and GPU
# sampling beside it. Run this *on the box*, from /workspace/warchest-v5.
#
#   tools/v5_stream.sh <tag> [seconds] [extra stream_bench args...]
#
# The checkpoint is a late one from a real 20-minute run, so the tree sizes are
# the mature ones the 30-minute goal actually has to sustain -- an early
# checkpoint measures the cheap opening workload and flatters everything.
set -euo pipefail

tag=$1
seconds=${2:-90}
shift 2 || shift 1 || true

ckpt=${V5_CKPT:-/workspace/warchest-engine/runs/v5_cardroute_s2_20m/snap_04.pt}
out=${V5_OUT:-/workspace/warchest-engine/runs}/$tag
mkdir -p "$out"

(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 1
    done
) >"$out/memory.csv" &
gpu_pid=$!

# %idle across all cores, once a second. The builders are the only heavy CPU
# user in a generation-only trace, so this says directly whether the host or
# the device is the limit.
(mpstat 1 "$((seconds + 40))" >"$out/cpu.txt" 2>&1 || true) &
cpu_pid=$!

finish() {
    kill "$gpu_pid" "$cpu_pid" 2>/dev/null || true
    wait "$gpu_pid" "$cpu_pid" 2>/dev/null || true
}
trap finish EXIT INT TERM

set -o pipefail
env WARCHEST_DIRECT=1 \
    WARCHEST_WAVE_LANES=${V5_LANES:-5} \
    WARCHEST_WAVE_WHALE_LANES=${V5_WHALE_LANES:-1} \
    WARCHEST_WAVE_ROWS=${V5_WAVE_ROWS:-196608} \
    WARCHEST_WAVE_JOBS=${V5_WAVE_JOBS:-256} \
    WARCHEST_WAVE_US=${V5_WAVE_US:-75000} \
    ${V5_ENV:-} \
    python -u train/stream_bench.py "$ckpt" \
    --seconds "$seconds" --devices ${V5_DEVICES:-0,1} --seed ${V5_SEED:-3} \
    --workers ${V5_WORKERS:-36} --actors ${V5_ACTORS:-128} \
    --inflight ${V5_INFLIGHT:-32} --chunk 1024 --report-every 10 \
    --cap-value 0 "$@" 2>&1 | tee "$out/stream.log"
