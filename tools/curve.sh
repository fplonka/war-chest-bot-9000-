#!/usr/bin/env bash
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export WARCHEST_BOX_DIR=/workspace/curve WARCHEST_BOX_LOCAL_DIR=$here
queue=$here/runs/curve/queue
results=$here/runs/curve/results.md
mkdir -p "$here/runs/curve"
[ -s "$results" ] || printf "| branch | 30 | 60 | 120 | 1h offset | slope per doubling |\n|---|---|---|---|---|---|\n" > "$results"

run() {
    local branch=$1 tree=$here/../curve-$branch
    shift
    git -C "$here" worktree add --force --detach "$tree" "$branch" &&
        "$tree/tools/box.sh" go --skip-ladder "$@"
    local status=$?
    git -C "$here" worktree remove --force "$tree"
    return $status
}

candidate() {
    local branch=$1
    "$here/tools/box.sh" "rm -rf runs/smoke_$branch" &&
        run "$branch" "out=smoke_$branch" minutes=2 warm_minutes=1 snapshot_every=1 seed=1 &&
        run "$branch" "out=${branch}_s1" minutes=120 snapshot_every=30 seed=1 &&
        run "$branch" "out=${branch}_s2" minutes=120 snapshot_every=30 seed=2 || return 1
    if [ "$branch" = master ]; then
        "$here/tools/box.sh" compare master_s1 master_s2 || return 1
    else
        for c in 1 2; do for b in 1 2; do
            "$here/tools/box.sh" compare "master_s$b" "${branch}_s$c" || return 1
        done; done
    fi
    python3 "$here/tools/curve.py" "$branch" >> "$results"
}

while :; do
    branch=$(head -1 "$queue" 2>/dev/null)
    if [ -z "$branch" ]; then
        sleep 60
        continue
    fi
    if ! "$here/tools/box.sh" true >/dev/null 2>&1; then
        echo "[$(date +%FT%T)] box unreachable, waiting"
        sleep 120
        continue
    fi
    echo "[$(date +%FT%T)] $branch"
    candidate "$branch" || {
        echo "| $branch | failed |" >> "$results"
        echo "[$(date +%FT%T)] $branch failed, holding the queue"
        sleep 120
        continue
    }
    tail -n +2 "$queue" > "$queue.tmp" && mv "$queue.tmp" "$queue"
done
