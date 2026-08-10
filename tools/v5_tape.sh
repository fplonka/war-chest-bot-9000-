#!/usr/bin/env bash
# One deterministic executor measurement. Same jobs every time, no game loop,
# no CPU tree building in the timed section -- so this isolates the GPU service
# from the self-play workload, which drifts as games advance.
#
#   tools/v5_tape.sh [seconds] [roots]      (run on the box, in /workspace/warchest-v5)
#
# Executor knobs come from the environment, so a sweep can vary one at a time.
set -euo pipefail

seconds=${1:-25}
roots=${2:-256}

exec env WARCHEST_DIRECT=1 \
    WARCHEST_WAVE_LANES=${V5_LANES:-5} \
    WARCHEST_WAVE_WHALE_LANES=${V5_WHALE_LANES:-1} \
    WARCHEST_WAVE_ROWS=${V5_WAVE_ROWS:-196608} \
    WARCHEST_WAVE_JOBS=${V5_WAVE_JOBS:-256} \
    WARCHEST_WAVE_US=${V5_WAVE_US:-75000} \
    WARCHEST_TAPE_DEVICES=${V5_DEVICES:-0,1} \
    WARCHEST_TAPE_PRODUCERS=${V5_PRODUCERS:-6} \
    WARCHEST_TAPE_QUEUE=${V5_QUEUE:-256} \
    ${V5_ENV:-} \
    ./engine/target/release/examples/wave_tape \
    tape/weights.bin tape/roots.bin "$roots" "$seconds"
