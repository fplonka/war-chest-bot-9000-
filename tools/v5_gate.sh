#!/usr/bin/env bash
# The real balanced-throughput gate: a training run on both cards, with the
# production ReBeL settings from docs/GPU_PERF_GOAL.md. Run on the box, from
# /workspace/warchest-v5.
#
#   tools/v5_gate.sh <tag> [total_minutes] [warm_minutes]
#
# Generation-only streams are a diagnostic; this is the number the goal is
# stated in, because it includes the trainer sharing GPU 1.
set -euo pipefail

tag=$1
minutes=${2:-10}
warm=${3:-5}
run_dir=/workspace/warchest-engine/runs/$tag
if [ -e "$run_dir" ]; then
    echo "run directory already exists: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"

(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 1
    done
) >"$run_dir/memory.csv" &
monitor=$!
(mpstat 1 "$(python3 -c "print(int($minutes*60)+120)")" >"$run_dir/cpu.txt" 2>&1 || true) &
cpu=$!
trap 'kill $monitor $cpu 2>/dev/null || true' EXIT INT TERM

export WARCHEST_DIRECT=1
export WARCHEST_WAVE_LANES=${V5_LANES:-8}
export WARCHEST_WAVE_WHALE_LANES=${V5_WHALE_LANES:-1}
export WARCHEST_WAVE_ROWS=${V5_WAVE_ROWS:-196608}
export WARCHEST_WAVE_JOBS=${V5_WAVE_JOBS:-256}
export WARCHEST_WAVE_US=${V5_WAVE_US:-75000}

set -o pipefail
python -u train/train.py \
    --minutes "$minutes" --warm-minutes "$warm" --warm-games 96 --random-draft \
    --depth 2 --iters 64 --cfr linear --warm 0 \
    --hidden 384 --head 0 --dg 64 --rank 64 --de 32 --nres 1 \
    --batch 1024 --train-gen-ratio 4 --lr 0.001 \
    --lr-decay-frac 0.33,0.67 \
    --recent-mix 0.5 --recent-frac 0.2 --policy 0 --aux 0 --mc-mix 0 \
    --explore 0.25 --temp 2 --eval-mix 0.5 \
    --cap-value 0.04 --anneal-frac 0.4 --snapshot-every 6 \
    --cap 2000000 --cfgs-per-row 48 \
    --gpu --gpu-devices 0,1 --gpu-workers "${V5_WORKERS:-36}" \
    --gpu-actors "${V5_ACTORS:-128}" --gpu-inflight "${V5_INFLIGHT:-32}" \
    --gpu-chunk 1024 --gpu-drain-seconds "${V5_DRAIN:-20}" \
    --gpu-publish-steps 16 --device cuda:1 \
    ${V5_INIT:+--init "$V5_INIT"} \
    --seed "${V5_SEED:-1}" --ladder-games "${V5_LADDER:-0}" \
    --out "$run_dir" 2>&1 | tee "$run_dir/train.log"
