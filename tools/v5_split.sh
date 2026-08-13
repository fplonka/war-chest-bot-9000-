#!/usr/bin/env bash
# Same total resources as one v5_stream.sh run -- 4,608 games, 10 GPU lanes,
# 36 builders -- but split across two processes, one per card. If the pair
# beats the single process, the ceiling is inside one process (a shared lock,
# context, or convoy), not the hardware.
set -euo pipefail

tag=$1
seconds=${2:-60}
ckpt=${V5_CKPT:-/workspace/warchest-engine/runs/v5_cardroute_s2_20m/snap_04.pt}
out=/workspace/warchest-engine/runs/$tag
mkdir -p "$out"

(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 1
    done
) >"$out/memory.csv" &
gpu_pid=$!
(mpstat 1 "$((seconds + 40))" >"$out/cpu.txt" 2>&1 || true) &
cpu_pid=$!
trap 'kill $gpu_pid $cpu_pid 2>/dev/null || true' EXIT INT TERM

for device in 0 1; do
    env WARCHEST_DIRECT=1 WARCHEST_WAVE_LANES=5 WARCHEST_WAVE_WHALE_LANES=1 \
        WARCHEST_WAVE_ROWS=196608 WARCHEST_WAVE_JOBS=256 WARCHEST_WAVE_US=75000 \
        python -u train/stream_bench.py "$ckpt" \
        --seconds "$seconds" --devices "$device" --seed "$((3 + device))" \
        --workers 18 --actors 128 --inflight 32 --chunk 1024 \
        --report-every 20 --cap-value 0 >"$out/stream$device.log" 2>&1 &
done
wait
grep -h prestop "$out"/stream0.log "$out"/stream1.log
