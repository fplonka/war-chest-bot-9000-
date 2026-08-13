#!/usr/bin/env bash
# Run a command on the Vast.ai GPU box (docs/GPU_PERF_GOAL.md) with the
# toolchain on PATH. Non-interactive ssh does not source the profile, so
# cargo, nvcc and the venv have to be put there by hand.
#
#   tools/box.sh <command...>          # run in /workspace/warchest-engine
#   tools/box.sh -bg <tag> <command...>  # detach; log to /workspace/logs/<tag>.log
#
# Override the endpoint with WARCHEST_BOX_HOST / WARCHEST_BOX_PORT.

set -euo pipefail

host=${WARCHEST_BOX_HOST:-184.144.224.246}
port=${WARCHEST_BOX_PORT:-40588}
key=${WARCHEST_BOX_KEY:-$HOME/.ssh/id_ed25519_warchest_vast}
remote=${WARCHEST_BOX_DIR:-/workspace/warchest-engine}

prelude="export PATH=/root/.cargo/bin:/usr/local/cuda/bin:\$PATH
export CARGO_BIN=/root/.cargo/bin/cargo
cd $remote"

if [[ ${1:-} == -bg ]]; then
    tag=$2
    shift 2
    body="$prelude
mkdir -p /workspace/logs
nohup setsid bash -lc $(printf '%q' "$prelude
$*") >/workspace/logs/$tag.log 2>&1 &
echo started $tag pid \$!"
else
    body="$prelude
$*"
fi

exec ssh -i "$key" -p "$port" -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 "root@$host" "$body"
