#!/usr/bin/env bash
# The GPU box, in one script: send the code, run something, bring results back.
#
#   tools/box.sh sync                     # push the working tree to the box
#   tools/box.sh <command...>             # run it there, output here
#   tools/box.sh -bg <tag> <command...>   # detach; log to /workspace/logs/<tag>.log
#   tools/box.sh pull                     # bring back reports, logs and ladders
#   tools/box.sh go dcfr                  # sync, build, run the experiment, pull
#
# `go` is the one to use. Everything an experiment needs happens in order and
# nothing is left to remember: the tree goes up, the extension is rebuilt, the
# arms run detached so a dropped ssh does not kill them, and the pages come
# back. Watch it with `tools/box.sh tail <name>`.
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
prelude="export PATH=/root/.cargo/bin:/usr/local/cuda/bin:\$PATH
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
tail)
    run_remote "tail -f /workspace/logs/${2:?usage: tail <tag>}.log"
    ;;
go)
    exp=${2:?usage: go <experiment> [extra exp.py args...]}
    shift 2
    "$0" sync
    run_remote "cd engine && maturin develop --release 2>&1 | tail -2"
    run_remote "mkdir -p /workspace/logs
nohup setsid bash -lc $(printf '%q' "$prelude
python train/exp.py run $exp $*") >/workspace/logs/$exp.log 2>&1 &
echo started $exp"
    echo "running. watch: tools/box.sh tail $exp   then: tools/box.sh pull"
    ;;
-bg)
    tag=${2:?usage: -bg <tag> <command...>}
    shift 2
    run_remote "mkdir -p /workspace/logs
nohup setsid bash -lc $(printf '%q' "$prelude
$*") >/workspace/logs/$tag.log 2>&1 &
echo started $tag"
    ;;
"")  sed -n '2,15p' "$0" ;;
*)   run_remote "$*" ;;
esac
