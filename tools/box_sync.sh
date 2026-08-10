#!/usr/bin/env bash
# Push the working tree to the GPU box (docs/GPU_PERF_GOAL.md). Source only:
# build outputs, runs and perf results stay where they are, and ownership is
# left as root's — copying the local uid across breaks git's safe-directory
# check on the box.

set -euo pipefail

host=${WARCHEST_BOX_HOST:-184.144.224.246}
port=${WARCHEST_BOX_PORT:-40588}
key=${WARCHEST_BOX_KEY:-$HOME/.ssh/id_ed25519_warchest_vast}
remote=${WARCHEST_BOX_DIR:-/workspace/warchest-engine}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# `--checksum --no-times`: rsync must not carry the local mtime across, or
# cargo's fingerprint sees a source file older than the object it produced and
# skips the rebuild — which silently benchmarks the previous binary.
exec rsync -rlpzvc --no-times --no-owner --no-group --delete \
    -e "ssh -i $key -p $port -o StrictHostKeyChecking=no" \
    --exclude '.git' \
    --exclude 'engine/target' \
    --exclude 'engine/target-prof' \
    --exclude 'perf' \
    --exclude 'runs' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'webui/node_modules' \
    "$here/" "root@$host:$remote/"
