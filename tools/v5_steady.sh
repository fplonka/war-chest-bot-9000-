#!/usr/bin/env bash
# A short training run in steady-state conditions. Run on the box, from
# /workspace/warchest-v5.
#
#   tools/v5_steady.sh <tag> [minutes]
#
# The thirty-minute golden run is the goal's metric but a terrible development
# loop: 32 minutes each and about +-5% run to run. The 180-second generation
# stream is fast but has no trainer, and the trainer is most of what limits the
# real run -- an optimizer step on the shared card costs 240 ms against 72-101
# uncontended, and the same Python thread is what drains finished solves.
#
# This is the middle: real train.py with the real trainer, but
#   * no Greedy warm-up, initialised from a late checkpoint, so the network is
#     already the one whose play produces expensive trees;
#   * --cap-value 0, so the horizon payoff is at its end state from the first
#     solve instead of annealing through the run.
# The games still start at move one, so the first ~90 seconds are a ramp. Read
# the result with tools/run_rate.py over a fixed span of *solves*, not over
# wall time: cost per solve grows as the games advance, so a faster build
# reaches the expensive positions sooner and its wall-time average converges
# back towards a slower build's.
set -euo pipefail

tag=$1
minutes=${2:-8}
shift 2 2>/dev/null || shift 1 || true
init=${V5_INIT:-/workspace/warchest-engine/runs/v5_cardroute_s2_20m/snap_04.pt}
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
(mpstat 1 "$(python3 -c "print(int($minutes*60)+60)")" >"$run_dir/cpu.txt" 2>&1 || true) &
cpu=$!
trap 'kill $monitor $cpu 2>/dev/null || true' EXIT INT TERM

export LD_PRELOAD=${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libjemalloc.so.2}
export WARCHEST_DIRECT=1
export WARCHEST_WAVE_LANES=${V5_LANES:-10}
export WARCHEST_WAVE_WHALE_LANES=${V5_WHALE_LANES:-1}
export WARCHEST_WAVE_ROWS=${V5_WAVE_ROWS:-196608}
export WARCHEST_WAVE_JOBS=${V5_WAVE_JOBS:-256}
export WARCHEST_WAVE_US=${V5_WAVE_US:-75000}
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

set -o pipefail
taskset -c 0-35 python -u train/train.py \
    --minutes "$minutes" --warm-minutes 0 --init "$init" --random-draft \
    --depth 2 --iters 64 --cfr linear --warm 0 \
    --hidden 384 --head 0 --dg 64 --rank 64 --de 32 --nres 1 \
    --batch 1024 --train-gen-ratio 4 --lr 0.001 \
    --lr-decay-frac 0.33,0.67 \
    --recent-mix 0.5 --recent-frac 0.2 --policy 0 --aux 0 --mc-mix 0 \
    --explore 0.25 --temp 2 --eval-mix 0.5 \
    --cap-value 0 --anneal-frac 0.4 --snapshot-every 60 \
    --cap 2000000 --cfgs-per-row 48 \
    --gpu --gpu-devices 0,1 --gpu-workers "${V5_WORKERS:-36}" \
    --gpu-actors "${V5_ACTORS:-128}" --gpu-inflight "${V5_INFLIGHT:-32}" \
    --gpu-chunk 1024 --gpu-drain-seconds "${V5_DRAIN:-40}" \
    --gpu-publish-steps 16 --device cuda:1 \
    --seed "${V5_SEED:-1}" --ladder-games 0 "$@" \
    --out "$run_dir" 2>&1 | tee "$run_dir/train.log"
