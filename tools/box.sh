#!/usr/bin/env bash
# The GPU box: send the code, run train.py, bring one run back when it finishes.
#
#   tools/box.sh go out=seat note="the idea"        train, queued behind other GPU jobs
#   tools/box.sh start m1 python tools/arena.py ...  any GPU job, same queue
#   tools/box.sh follow m1
#   tools/box.sh pull seat
#   tools/box.sh setup                             a fresh vast.ai pytorch image
#   tools/box.sh sync
#   tools/box.sh build
#   tools/box.sh <command...>
set -euo pipefail

host=${WARCHEST_BOX_HOST:-ssh3.vast.ai}
port=${WARCHEST_BOX_PORT:-10570}
key=${WARCHEST_BOX_KEY:-$HOME/.ssh/id_ed25519_warchest_vast}
remote=${WARCHEST_BOX_DIR:-/workspace/warchest-engine}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

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
    name=${2:?usage: pull <run>}
    mkdir -p "$here/runs/$name"
    # `*.tmp` is a live run writing log.json atomically; rsync would list it,
    # find it replaced, and exit 24 mid-run.
    rsync -az -e "ssh ${ssh_opts[*]}" --exclude '*.tmp' \
        "root@$host:$remote/runs/$name/" "$here/runs/$name/"
    echo "pulled $name"
    ;;
build)
    "$0" start build bash -c "$build_script"
    "$0" follow build
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
        # A queued job has no run directory until it holds the lock.
        [ -n "${3:-}" ] && { "$0" pull "$3" >/dev/null 2>&1 || true; }
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
start)
    # start <tag> <command...>: the command runs detached on the box, queued
    # behind every other GPU job by one lock, with its script, log, pid and
    # exit code under /workspace/logs/<tag>.*. `follow <tag>` waits on it.
    tag=${2:?usage: start <tag> <command...>}
    shift 2
    run_remote "mkdir -p /workspace/logs
rm -f /workspace/logs/$tag.pid /workspace/logs/$tag.exit
cat > /workspace/logs/$tag.sh <<'EOS'
$prelude
$(printf '%q ' "$@")
EOS
nohup setsid bash -c 'echo \$\$ > /workspace/logs/$tag.pid
flock /workspace/gpu.lock bash /workspace/logs/$tag.sh
echo \$? > /workspace/logs/$tag.exit' >/workspace/logs/$tag.log 2>&1 &
echo started $tag"
    ;;
go)
    # The build runs inside the job, under the lock: a cargo build beside a
    # measured run would take its cores.
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
