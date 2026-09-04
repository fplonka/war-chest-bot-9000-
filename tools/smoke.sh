#!/usr/bin/env bash
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export WARCHEST_BOX_DIR=/workspace/curve WARCHEST_BOX_LOCAL_DIR=$here

for branch in "$@"; do
    tree=$here/../smoke-$branch
    echo "[$(date +%FT%T)] $branch"
    git -C "$here" worktree add --force --detach "$tree" "$branch" &&
        "$tree/tools/box.sh" go --skip-ladder "out=smoke_$branch" \
            minutes=2 warm_minutes=1 snapshot_every=1 seed=1
    echo "[$(date +%FT%T)] $branch exit $?"
    git -C "$here" worktree remove --force "$tree"
done
