#!/usr/bin/env bash
# Run a command in the v5 checkout on the GPU box, with cargo, nvcc and the
# python venv on PATH (a non-interactive ssh does not source the profile).
#
#   tools/v5.sh <command...>
#   tools/v5.sh -bg <tag> <command...>   # detach, log to /workspace/logs/<tag>.log
set -euo pipefail

host=${WARCHEST_BOX_HOST:-184.144.224.246}
port=${WARCHEST_BOX_PORT:-40588}
key=${WARCHEST_BOX_KEY:-$HOME/.ssh/id_ed25519_warchest_vast}
remote=${WARCHEST_V5_DIR:-/workspace/warchest-v5}

prelude="export PATH=/root/.cargo/bin:/usr/local/cuda/bin:\$PATH
source /venv/main/bin/activate
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
    -o ServerAliveInterval=30 -o BatchMode=yes "root@$host" "$body"
