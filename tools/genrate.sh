#!/usr/bin/env bash
# The only rate that counts: what `train.py` generates.
#
# The bench solves a fixed corpus with the cards to itself. A run trains on
# `cuda:1` while generation uses both cards, publishes weights mid-run, and its
# position distribution moves as play improves. Six minutes is enough to read
# the generation rate and nothing here is meant to train anything.
set -uo pipefail
cd /workspace/warchest-engine
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
source /venv/main/bin/activate 2>/dev/null || true
mkdir -p /workspace/logs
OUT=${OUT:-ratecheck}
rm -rf "runs/$OUT"
setsid nohup python train/train.py "out=$OUT" minutes=6 warm_minutes=1 \
    warm_games=0 ladder_games=0 "note=generation rate only" \
    > /workspace/logs/genrate.log 2>&1 &
wait $!
grep -E "GT-CFR|summary" /workspace/logs/genrate.log | tail -20
