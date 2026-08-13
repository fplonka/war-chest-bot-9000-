#!/usr/bin/env bash
# The GPU box: send the code, run train.py, bring reports back.
#
#   tools/box.sh go out=seat note="the idea"
#   tools/box.sh follow run
#   tools/box.sh pull
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
        --exclude runs --exclude target --exclude .venv --exclude papers \
        --exclude __pycache__ --exclude .git --exclude data \
        "$here/" "root@$host:$remote/"
    sha=$(git -C "$here" rev-parse --short=7 HEAD)
    git -C "$here" diff-index --quiet HEAD -- . ':!runs' || sha="${sha}+dirty"
    echo "$sha" | ssh "${ssh_opts[@]}" "root@$host" "cat > $remote/.gitsha"
    echo "synced -> $host:$remote"
    ;;
pull)
    mkdir -p "$here/runs"
    rsync -az -e "ssh ${ssh_opts[*]}" \
        --include '*/' --include '*.html' --include 'plotly.min.js' --include 'log.json' \
        --include 'ladder.json' --include 'config.json' --include 'NOTES.md' \
        --include 'train.log' \
        --exclude '*' "root@$host:$remote/runs/" "$here/runs/"
    echo "pulled reports into $here/runs"
    ;;
build)
    run_remote "find engine/src engine/tests engine/examples -type f -exec touch {} +
cd engine && maturin develop --release --features python,gpu 2>&1 | tail -2"
    ;;
follow)
    tag=${2:?usage: follow <tag>}
    while "$0" pull >/dev/null \
        && run_remote "kill -0 \$(cat /workspace/logs/$tag.pid) 2>/dev/null" >/dev/null 2>&1; do
        run_remote "tail -1 /workspace/logs/$tag.log" | tail -1
        sleep "${WARCHEST_BOX_POLL:-60}"
    done
    "$0" pull
    if ! run_remote "grep -qx 0 /workspace/logs/$tag.exit"; then
        echo "job $tag failed; tail /workspace/logs/$tag.log"
        run_remote "tail -20 /workspace/logs/$tag.log" | tail -20
        exit 1
    fi
    ;;
go)
    shift
    "$0" sync
    "$0" build
    cmd=$(printf '%q ' python train/train.py "$@")
    run_remote "mkdir -p /workspace/logs
nohup setsid bash -lc $(printf '%q' "$prelude
echo \$\$ > /workspace/logs/run.pid
$cmd
echo \$? > /workspace/logs/run.exit") >/workspace/logs/run.log 2>&1 &
echo started run"
    "$0" follow run
    ;;
"")  sed -n '2,10p' "$0" ;;
*)   run_remote "$*" ;;
esac
