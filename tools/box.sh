#!/usr/bin/env bash
# The GPU box: send the code, run train.py, bring one run back when it finishes.
#
#   tools/box.sh go out=seat note="the idea"
#   tools/box.sh follow run
#   tools/box.sh pull seat
#   tools/box.sh sync
#   tools/box.sh build
#   tools/box.sh <command...>
set -euo pipefail

host=${WARCHEST_BOX_HOST:-184.144.224.246}
port=${WARCHEST_BOX_PORT:-40588}
key=${WARCHEST_BOX_KEY:-$HOME/.ssh/id_ed25519_warchest_vast}
remote=${WARCHEST_BOX_DIR:-/workspace/warchest-engine}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=no -o ServerAliveInterval=30)
prelude="export PATH=/root/.cargo/bin:/usr/local/cuda/bin:\$PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
source /venv/main/bin/activate 2>/dev/null || true
cd $remote"

run_remote() { ssh "${ssh_opts[@]}" "root@$host" "$prelude
$*"; }

case "${1:-}" in
sync)
    rsync -az --delete -e "ssh ${ssh_opts[*]}" \
        --exclude runs --exclude bots --exclude arena --exclude suites \
        --exclude target --exclude .venv --exclude papers \
        --exclude __pycache__ --exclude .git --exclude data \
        "$here/" "root@$host:$remote/"
    sha=$(git -C "$here" rev-parse --short=7 HEAD)
    git -C "$here" diff-index --quiet HEAD -- . ':!runs' || sha="${sha}+dirty"
    echo "$sha" | ssh "${ssh_opts[@]}" "root@$host" "cat > $remote/.gitsha"
    echo "synced -> $host:$remote"
    ;;
pull)
    name=${2:?usage: pull <run>}
    mkdir -p "$here/runs/$name"
    # `*.tmp` is a live run writing log.json atomically; rsync would list it,
    # find it replaced, and exit 24 mid-run.
    rsync -az -e "ssh ${ssh_opts[*]}" --exclude '*.pt' --exclude '*.tmp' \
        "root@$host:$remote/runs/$name/" "$here/runs/$name/"
    echo "pulled $name"
    ;;
build)
    run_remote "find engine/src engine/tests engine/examples -type f -exec touch {} +
cd engine && maturin develop --release --features python,gpu 2>&1 | tail -2
cargo build --release --bin bot 2>&1 | tail -1"
    ;;
follow)
    tag=${2:?usage: follow <tag> [run]}
    for _ in $(seq 1 40); do
        if run_remote "test -s /workspace/logs/$tag.pid"; then
            break
        fi
        sleep 0.5
    done
    while run_remote "kill -0 \$(cat /workspace/logs/$tag.pid) 2>/dev/null" >/dev/null 2>&1; do
        [ -n "${3:-}" ] && "$0" pull "$3" >/dev/null
        run_remote "tail -1 /workspace/logs/$tag.log" | tail -1
        sleep "${WARCHEST_BOX_POLL:-60}"
    done
    [ -n "${3:-}" ] && "$0" pull "$3"
    if ! run_remote "grep -qx 0 /workspace/logs/$tag.exit"; then
        echo "JOB_DONE tag=$tag failed"
        run_remote "tail -20 /workspace/logs/$tag.log" | tail -20
        exit 1
    fi
    echo "JOB_DONE tag=$tag ok"
    ;;
go)
    shift
    out=
    for a in "$@"; do
        case "$a" in out=*) out=${a#out=} ;; esac
    done
    [ -n "$out" ] || { echo "go needs out=<name>" >&2; exit 1; }
    "$0" sync
    "$0" build
    cmd=$(printf '%q ' python train/train.py "$@")
    run_remote "mkdir -p /workspace/logs
rm -f /workspace/logs/run.pid /workspace/logs/run.exit
nohup setsid bash -lc $(printf '%q' "$prelude
echo \$\$ > /workspace/logs/run.pid
$cmd
echo \$? > /workspace/logs/run.exit") >/workspace/logs/run.log 2>&1 &
echo started run"
    "$0" follow run "$out"
    ;;
"")  sed -n '2,9p' "$0" ;;
*)   run_remote "$*" ;;
esac
