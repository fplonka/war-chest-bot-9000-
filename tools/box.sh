#!/usr/bin/env bash
# The GPU box: send the code, run train.py, bring one run back when it finishes.
#
#   tools/box.sh go out=seat note="the idea"        train, queued behind other GPU jobs
#   tools/box.sh start m1 python tools/arena.py ...  any GPU job, same queue
#   tools/box.sh follow m1
#   tools/box.sh kill m1                            the job and everything it spawned
#   tools/box.sh pull [run]
#   tools/box.sh setup                             a fresh vast.ai pytorch image
#   tools/box.sh sync
#   tools/box.sh build
#   tools/box.sh <command...>
set -euo pipefail

host=${WARCHEST_BOX_HOST:-ssh1.vast.ai}
port=${WARCHEST_BOX_PORT:-26778}
key=${WARCHEST_BOX_KEY:-$HOME/.ssh/id_ed25519_warchest_vast}
remote=${WARCHEST_BOX_DIR:-/workspace/warchest-engine}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
local_dir=${WARCHEST_BOX_LOCAL_DIR:-$here}

ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=no -o ServerAliveInterval=30)
# jemalloc and the image's virtualenv are both nice to have and neither is
# guaranteed by the image, so neither is allowed to stop a command.
prelude="export PATH=/root/.cargo/bin:/usr/local/cuda/bin:\$PATH
# maturin insists on knowing which environment it is installing into, and the
# image's python is a conda root rather than a virtualenv.
export CONDA_PREFIX=\${CONDA_PREFIX:-/opt/conda}
[ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ] && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
source /venv/main/bin/activate 2>/dev/null || true
mkdir -p $remote && cd $remote"

run_remote() { ssh "${ssh_opts[@]}" "root@$host" "$prelude
$*"; }

# No pipe on the build itself: a pipeline exits with `tail`'s status, so a
# failed build used to look like a success and leave the old module in
# place — which is how a run started without the `gpu` feature.
build_script="find engine/src engine/tests engine/examples -type f -exec touch {} +
cd engine
maturin develop --release --features python,gpu >/tmp/maturin.log 2>&1 || { tail -40 /tmp/maturin.log; exit 1; }
tail -2 /tmp/maturin.log
cargo build --release --features gpu --bin bot >/tmp/bot.log 2>&1 || { tail -40 /tmp/bot.log; exit 1; }
tail -1 /tmp/bot.log"

case "${1:-}" in
setup)
    # A fresh `pytorch/pytorch:*-devel` image: the allocator, the Rust
    # toolchain, and the two Python packages the trainer needs beyond torch.
    run_remote "apt-get update -qq && apt-get install -y -qq libjemalloc2 rsync >/dev/null
[ -x /root/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
pip install -q maturin numpy
cargo --version && nvcc --version | tail -1 && python -c 'import torch; print(torch.__version__, torch.cuda.device_count(), \"gpus\")'"
    ;;
sync)
    rsync -az --delete -e "ssh ${ssh_opts[*]}" \
        --exclude runs --exclude bots --exclude arena --exclude suites \
        --exclude target --exclude .venv --exclude papers \
        --exclude __pycache__ --exclude .git --exclude data \
        "$here/" "root@$host:$remote/"
    # One runs/ per box: every checkout shares warchest-engine's, so the
    # results ledger and the dashboard have one place to look.
    [ "$remote" = /workspace/warchest-engine ] ||
        run_remote "[ -e runs ] || ln -s /workspace/warchest-engine/runs runs"
    sha=$(git -C "$here" rev-parse --short=7 HEAD)
    git -C "$here" diff-index --quiet HEAD -- . ':!runs' || sha="${sha}+dirty"
    echo "$sha" | ssh "${ssh_opts[@]}" "root@$host" "cat > $remote/.gitsha"
    echo "synced -> $host:$remote"
    ;;
pull)
    name=${2:-}
    filters=(--exclude '*.tmp')
    if [ -z "$name" ]; then
        filters+=(--include '*/' --include '*.html' --include 'plotly.min.js'
                  --include 'log.json' --include 'ladder*.json' --include 'config.json'
                  --include 'NOTES.md' --include 'train.log' --exclude '*')
    fi
    # `*.tmp` is a live run writing log.json atomically; rsync would list it,
    # find it replaced, and exit 24 mid-run.
    mkdir -p "$local_dir/runs${name:+/$name}"
    rsync -az -e "ssh ${ssh_opts[*]}" "${filters[@]}" \
        "root@$host:$remote/runs/$name/" "$local_dir/runs${name:+/$name}/"
    echo "pulled ${name:-runs}"
    ;;
build)
    "$0" start build bash -c "$build_script"
    "$0" follow build
    ;;
follow)
    tag=${2:?usage: follow <tag> [run]}
    if [ -n "${3:-}" ]; then
        (while :; do
            "$0" pull "$3" >/dev/null 2>&1 || true
            sleep "${WARCHEST_BOX_POLL:-60}"
        done) &
        pull_pid=$!
        trap 'kill "$pull_pid" 2>/dev/null || true; wait "$pull_pid" 2>/dev/null || true' EXIT
    fi
    status=0
    run_remote "while [ ! -s /workspace/logs/$tag.pid ] || kill -0 \$(cat /workspace/logs/$tag.pid) 2>/dev/null; do
    tail -1 /workspace/logs/$tag.log 2>/dev/null || true
    sleep \${WARCHEST_BOX_POLL:-60}
done
cat /workspace/logs/$tag.exit
grep -qx 0 /workspace/logs/$tag.exit || { tail -20 /workspace/logs/$tag.log; exit 1; }" || status=$?
    [ -n "${3:-}" ] && "$0" pull "$3"
    [ "$status" -eq 0 ] && echo "JOB_DONE tag=$tag ok" || {
        echo "JOB_DONE tag=$tag failed"
        exit 1
    }
    ;;
start)
    # start <tag> <command...>: the command runs detached on the box, queued
    # behind every other GPU job in order of arrival by a ticket in
    # /workspace/queue. Its script, log, pid and exit code live under
    # /workspace/logs/<tag>.*. `follow <tag>` waits.
    tag=${2:?usage: start <tag> <command...>}
    shift 2
    run_remote "mkdir -p /workspace/logs /workspace/queue
rm -f /workspace/logs/$tag.pid /workspace/logs/$tag.exit
echo $remote > /workspace/logs/$tag.owner
cat > /workspace/logs/$tag.sh <<'EOS'
echo \$\$ > /workspace/logs/$tag.pid
ticket=/workspace/queue/\$(date +%s%N)-$tag
touch \$ticket
trap 'rm -f \$ticket' EXIT
while [ \"\$(ls /workspace/queue | head -1)\" != \"\$(basename \$ticket)\" ]; do sleep 2; done
$prelude
$(printf '%q ' "$@")
status=\$?
echo \$status > /workspace/logs/$tag.exit
exit \$status
EOS
nohup setsid bash /workspace/logs/$tag.sh >/workspace/logs/$tag.log 2>&1 &
echo started $tag"
    ;;
kill)
    # The job's pid is its process-group leader (setsid), so this takes the
    # whole tree: an orphaned test binary once held the queue for an hour.
    # Only the checkout that started a job may kill it: workers killed each
    # other's queued runs and a driver match by "tidying" the queue.
    tag=${2:?usage: kill <tag>}
    run_remote "[ \"\$(cat /workspace/logs/$tag.owner 2>/dev/null)\" = $remote ] || { echo \"$tag was not started from $remote; not killed\"; exit 1; }
kill -- -\$(cat /workspace/logs/$tag.pid) && echo killed $tag"
    ;;
go)
    # The build runs inside the queued job, so a cargo build beside a measured
    # run cannot take its cores.
    shift
    out=
    for a in "$@"; do
        case "$a" in out=*) out=${a#out=} ;; esac
    done
    [ -n "$out" ] || { echo "go needs out=<name>" >&2; exit 1; }
    "$0" sync
    "$0" start "$out" bash -c "($build_script) && exec python train/train.py $(printf '%q ' "$@")"
    "$0" follow "$out" "$out"
    ;;
"")  sed -n '2,9p' "$0" ;;
*)   run_remote "$*" ;;
esac
