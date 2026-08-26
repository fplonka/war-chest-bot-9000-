#!/usr/bin/env bash
# The only rate that counts: what `train.py` generates.
#
# The bench solves a fixed corpus with the cards to itself. A run trains on
# `cuda:1` while generation uses both cards, publishes weights mid-run, and its
# position distribution moves as play improves. Six minutes is enough to read
# the generation rate and nothing here is meant to train anything.
#
# The two bounds are sampled beside it: card utilisation, and the host RSS the
# admission level is set against.
set -uo pipefail
cd /workspace/warchest-engine
# box.sh's prelude, so the rate read here is the rate a run gets.
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CONDA_PREFIX=${CONDA_PREFIX:-/opt/conda}
[ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ] && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
source /venv/main/bin/activate 2>/dev/null || true

OUT=${OUT:-ratecheck}
LOGS=/workspace/logs
mkdir -p "$LOGS"
rm -rf "runs/$OUT"

python train/train.py "out=$OUT" minutes=6 snapshot_every=30 \
    "note=generation rate only" > "$LOGS/genrate.log" 2>&1 &
pid=$!

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv -l 5 > "$LOGS/genrate_gpu.csv" 2>&1 &
gpu=$!
while kill -0 "$pid" 2>/dev/null; do
    echo "$(date +%s) $(ps -o rss= -p "$pid")"
    sleep 5
done > "$LOGS/genrate_rss.txt" &
rss=$!

wait "$pid"
status=$?
kill "$gpu" "$rss" 2>/dev/null
grep -E "GT-CFR|summary" "$LOGS/genrate.log" | tail -20
exit "$status"
