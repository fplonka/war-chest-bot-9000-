#!/usr/bin/env bash
# The whole post-run sequence in one go: rebuild, gate on the CUDA tests, A/B
# the readout formulation on the mature stream, then rate the golden run's
# snapshots. Run on the box, from /workspace/warchest-v5.
set -uo pipefail

cd /workspace/warchest-v5

echo "=== build and correctness gate ==="
(cd engine && cargo test --release --features gpu --lib -- --test-threads=1 2>&1 | tail -4)
(cd engine && maturin develop --release --features python,gpu 2>&1 | tail -1)

# Interleaved, because a single pair on this box is not separable from drift.
for rep in 1 2; do
    echo "=== rep $rep: readout one-config-per-lane ==="
    V5_LANES=10 bash tools/v5_stream.sh "v5_ro_lane_$rep" 120 >/dev/null 2>&1
    grep prestop "/workspace/warchest-engine/runs/v5_ro_lane_$rep/stream.log"
    echo "=== rep $rep: readout warp-reduction (control) ==="
    V5_LANES=10 V5_ENV=WARCHEST_READOUT_WARP=1 \
        bash tools/v5_stream.sh "v5_ro_warp_$rep" 120 >/dev/null 2>&1
    grep prestop "/workspace/warchest-engine/runs/v5_ro_warp_$rep/stream.log"
done

echo "=== ladder over the golden run's snapshots ==="
python -u train/ladder.py /workspace/warchest-engine/runs/gpu_golden8 \
    --games 30 --random-draft --gpu --refs greedy,random \
    2>&1 | tail -40
echo NEXT_DONE
