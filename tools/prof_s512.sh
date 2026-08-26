#!/usr/bin/env bash
# Real-run nsys window: 3 min greedy warm, then SoG. Delay 210 s so the 120 s
# trace sits in SoG; train.py --minutes=8 covers the window and exits itself.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
source /venv/main/bin/activate 2>/dev/null || true

if pgrep -f 'python(3)? train/train.py' >/dev/null; then
    echo "train.py already running" >&2
    pgrep -af 'python(3)? train/train.py' >&2
    exit 1
fi

OUT=/workspace/logs/prof
NSYS_OUT=$OUT/train_s512
RUN=runs/prof_exp
mkdir -p "$OUT"
rm -rf "$RUN" "$NSYS_OUT".nsys-rep "$NSYS_OUT".sqlite "$NSYS_OUT".qdstrm
rm -f "$OUT"/summary.txt "$OUT"/gaps.txt "$OUT"/nsys.log "$OUT"/export.log

if ! command -v py-spy >/dev/null 2>&1; then
    pip install --quiet py-spy || true
fi

bash tools/host_sample.sh "$OUT" 225 >"$OUT/host_sample.log" 2>&1 &
sample_pid=$!

# --kill none: duration ends the trace; train.py keeps going until --minutes.
# --sample/--cpuctxsw none: CUPTI only. Hardware metrics need admin and are off.
set +e
/usr/local/cuda/bin/nsys profile \
    -o "$NSYS_OUT" \
    --force-overwrite true \
    --trace=cuda \
    --sample=none \
    --cpuctxsw=none \
    --gpu-metrics-devices=none \
    --delay=210 \
    --duration=120 \
    --kill=none \
    python train/train.py \
        out="$RUN" minutes=8 snapshot_every=30 \
        s=512 c=8 round_batch=8 gen_workers=36 gen_solves=8 \
        device=cuda:1 gen_devices=0,1 batch=256 target_every=1 \
        note="nsys 210s delay + 120s window after 3min warm, s=512" \
    >"$OUT/train.log" 2>&1
nsys_rc=$?
set -e
wait "$sample_pid" || true
echo "nsys_exit=$nsys_rc" | tee "$OUT/nsys.exit"

if [ ! -s "$NSYS_OUT.nsys-rep" ]; then
    echo "missing nsys-rep" >&2
    tail -50 "$OUT/train.log" >&2
    exit 1
fi

/usr/local/cuda/bin/nsys export --type sqlite --force-overwrite true \
    -o "$NSYS_OUT.sqlite" "$NSYS_OUT.nsys-rep" >"$OUT/export.log" 2>&1

python tools/nsys_summary.py "$NSYS_OUT.sqlite" >"$OUT/summary.txt"
python tools/gaps.py "$NSYS_OUT.sqlite" >"$OUT/gaps.txt" || true
ls -lh "$OUT" | tee "$OUT/ls.txt"
echo DONE
