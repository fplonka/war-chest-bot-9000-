#!/usr/bin/env bash
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export WARCHEST_BOX_DIR=/workspace/curve WARCHEST_BOX_LOCAL_DIR=$here

for branch in "$@"; do
    tree=$here/../smoke-$branch
    echo "[$(date +%FT%T)] $branch"
    git -C "$here" worktree add --force --detach "$tree" "$branch" &&
        "$tree/tools/box.sh" "rm -rf runs/smoke_$branch" &&
        "$tree/tools/box.sh" go --skip-ladder "out=smoke_$branch" \
            minutes=2 warm_minutes=1 snapshot_every=1 seed=1
    status=$?
    git -C "$here" worktree remove --force "$tree"
    [ "$status" -eq 0 ] || echo "SMOKE FAILED $branch exit $status"
done
echo "SMOKE DONE"
