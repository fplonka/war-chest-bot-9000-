#!/usr/bin/env bash
# Push the working tree to the v5 checkout on the GPU box
# (/workspace/warchest-v5). No --delete: the box keeps tape/, .venv/ and
# engine/target/ that are not in the local tree.
#
# --checksum --no-times: rsync must not carry the local mtime across, or
# cargo's fingerprint sees a source file older than its object and skips the
# rebuild, which silently benchmarks the previous binary.
set -euo pipefail

host=${WARCHEST_BOX_HOST:-184.144.224.246}
port=${WARCHEST_BOX_PORT:-40588}
key=${WARCHEST_BOX_KEY:-$HOME/.ssh/id_ed25519_warchest_vast}
remote=${WARCHEST_V5_DIR:-/workspace/warchest-v5}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

exec rsync -rlpzc --no-times --no-owner --no-group \
    -e "ssh -i $key -p $port -o StrictHostKeyChecking=no -o BatchMode=yes" \
    --exclude '.git' \
    --exclude 'engine/target' \
    --exclude 'engine/target-prof' \
    --exclude 'perf' \
    --exclude 'runs' \
    --exclude 'papers' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'webui/node_modules' \
    "$@" \
    "$here/" "root@$host:$remote/"
