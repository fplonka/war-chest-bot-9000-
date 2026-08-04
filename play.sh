#!/usr/bin/env bash
# One-liner: build the extension (if needed) and open the browser UI against
# the newest trained checkpoint.
#
#   ./play.sh                 # newest runs/*/ckpt_final.pt
#   ./play.sh --ckpt runs/cfgvalue01/ckpt_final.pt
#   ./play.sh --port 9000
#
# Extra arguments are passed through to webui/play.py.
set -euo pipefail
cd "$(dirname "$0")"

# A venv to build into: this checkout's own if it has one, otherwise the main
# checkout's (the repo's venv is gitignored and not copied into worktrees).
VENV=""
for cand in "$PWD/.venv" "$PWD/../warchest-engine/.venv"; do
  if [ -x "$cand/bin/maturin" ]; then VENV="$cand"; break; fi
done
if [ -z "$VENV" ]; then
  echo "no venv found (.venv or ../warchest-engine/.venv). Create one with:" >&2
  echo "  uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install torch numpy maturin" >&2
  exit 1
fi

# Rebuild the extension so the installed module matches this checkout. The
# build is incremental; when nothing changed it only relinks.
(cd engine && VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release --quiet 2>/dev/null \
  || { echo "engine build failed — see the error above" >&2; exit 1; })

exec env VIRTUAL_ENV="$VENV" "$VENV/bin/python" webui/play.py "$@"
