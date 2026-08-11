#!/usr/bin/env bash
# The GPU box, in one script: send the code, run something, bring results back.
#
#   tools/box.sh sync                     # push the working tree to the box
#   tools/box.sh build                    # rebuild the extension there, with CUDA
#   tools/box.sh <command...>             # run it there, output here
#   tools/box.sh -bg <tag> <command...>   # detach; log to /workspace/logs/<tag>.log
#   tools/box.sh pull                     # bring back reports, logs and ladders
#   tools/box.sh follow dcfr              # watch an experiment, pulling as it goes
#   tools/box.sh go dcfr                  # sync, build, run the experiment, pull
#
# `go` is the one to use. Everything an experiment needs happens in order and
# nothing is left to remember: the tree goes up, the extension is rebuilt, the
# arms run detached so a dropped ssh does not kill them, and `follow` brings
# each run's page back as it finishes. Ctrl-C is safe -- the run is detached,
# so following it again, or a plain `pull`, picks up where this left off.
#
# A non-interactive ssh does not source the profile, so cargo, nvcc and the venv
# are put on PATH by hand.
set -euo pipefail

host=${WARCHEST_BOX_HOST:-184.144.224.246}
port=${WARCHEST_BOX_PORT:-40588}
key=${WARCHEST_BOX_KEY:-$HOME/.ssh/id_ed25519_warchest_vast}
remote=${WARCHEST_BOX_DIR:-/workspace/warchest-v5}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=no -o ServerAliveInterval=30)
# jemalloc is worth ~4% of generation throughput (docs/GPU_ARCHITECTURE.md);
# glibc malloc is what the 36 builder threads contend on. mimalloc was tried
# and exhausts the static TLS block, after which torch cannot load libgomp.
prelude="export PATH=/root/.cargo/bin:/usr/local/cuda/bin:\$PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
source /venv/main/bin/activate 2>/dev/null || true
cd $remote"

run_remote() { ssh "${ssh_opts[@]}" "root@$host" "$prelude
$*"; }

case "${1:-}" in
sync)
    # Source only. Checkpoints and dumps are large, live on the box, and are
    # never the thing that needs to travel upward.
    rsync -az --delete -e "ssh ${ssh_opts[*]}" \
        --exclude runs --exclude target --exclude .venv --exclude papers \
        --exclude __pycache__ --exclude .git --exclude data \
        "$here/" "root@$host:$remote/"
    # .git is not sent, so leave the sha behind: a run there still records the
    # commit it came from.
    git -C "$here" describe --always --dirty --abbrev=7 \
        | ssh "${ssh_opts[@]}" "root@$host" "cat > $remote/.gitsha"
    echo "synced -> $host:$remote"
    ;;
pull)
    # Reports, logs and ladders: everything needed to read a result, and none
    # of the weights.
    mkdir -p "$here/runs"
    rsync -az -e "ssh ${ssh_opts[*]}" \
        --include '*/' --include '*.html' --include 'log.json' \
        --include 'ladder.json' --include 'config.json' --include 'NOTES.md' \
        --exclude '*' "root@$host:$remote/runs/" "$here/runs/"
    echo "pulled reports into $here/runs"
    ;;
build)
    # Without `gpu` the extension has no gpu_start and every run on this box
    # dies at startup. `python` must be repeated: --features replaces the
    # pyproject list rather than adding to it, and without pyo3 maturin builds
    # a cffi module instead.
    run_remote "cd engine && maturin develop --release --features python,gpu 2>&1 | tail -2"
    ;;
tail)
    run_remote "tail -f /workspace/logs/${2:?usage: tail <tag>}.log"
    ;;
follow)
    # A run writes its report.html when it ends, so poll: pull whatever exists,
    # print where the work has got to, and stop when its process is gone. One
    # ssh a minute costs nothing against runs measured in tens of minutes, and
    # tailing instead would hold a connection open for hours to learn the same
    # thing.
    tag=${2:?usage: follow <tag>}
    while "$0" pull >/dev/null \
        && run_remote "kill -0 \$(cat /workspace/logs/$tag.pid) 2>/dev/null" >/dev/null 2>&1; do
        # The box's login files greet every ssh, so keep the last line only.
        run_remote "tail -1 /workspace/logs/$tag.log" | tail -1
        sleep "${WARCHEST_BOX_POLL:-60}"
    done
    "$0" pull
    ;;
go)
    exp=${2:?usage: go <experiment> [extra exp.py args...]}
    shift 2
    "$0" sync
    "$0" build
    "$0" -bg "$exp" "python train/exp.py run $exp $*"
    "$0" follow "$exp"
    ;;
-bg)
    # The job records its own pid, which is what `follow` waits on. `setsid`
    # would otherwise have exited by the time anyone looked, and matching the
    # command line instead only works for jobs whose tag appears in it.
    tag=${2:?usage: -bg <tag> <command...>}
    shift 2
    run_remote "mkdir -p /workspace/logs
nohup setsid bash -lc $(printf '%q' "$prelude
echo \$\$ > /workspace/logs/$tag.pid
$*") >/workspace/logs/$tag.log 2>&1 &
echo started $tag"
    ;;
"")  sed -n '2,16p' "$0" ;;
*)   run_remote "$*" ;;
esac
