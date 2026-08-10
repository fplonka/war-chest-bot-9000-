#!/usr/bin/env bash
# Reproducible CUDA edit -> gate -> measure loop. Run this on the target GPU
# host from anywhere inside the checkout:
#
#   tools/gpu_iter.sh baseline       # build, gate, freeze reference binaries
#   tools/gpu_iter.sh quick          # build, phase oracle, trained + zero runs
#   tools/gpu_iter.sh compare        # interleaved reference/candidate A/B
#   tools/gpu_iter.sh profile        # Nsight Systems timeline + text reports
#   tools/gpu_iter.sh all            # full gate, comparison, and profile
#
# Results land under perf/gpu/<timestamp>-<mode>. The benchmark binary is kept
# separate from the output directories so later source builds cannot silently
# replace the reference side of an A/B.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
engine_dir="$repo_root/engine"
mode=${1:-quick}
stamp=${GPU_PERF_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$mode}
out_root=${GPU_PERF_OUT:-"$repo_root/perf/gpu"}
out_dir="$out_root/$stamp"
baseline_dir=${GPU_PERF_BASELINE_DIR:-"$out_root/baseline"}
cargo_bin=${CARGO_BIN:-cargo}
if cargo_path=$(command -v -- "$cargo_bin" 2>/dev/null); then
    cargo_bin=$cargo_path
elif [[ ! -x "$cargo_bin" ]]; then
    printf 'cargo is not executable: %s (set CARGO_BIN to its absolute path)\n' \
        "$cargo_bin" >&2
    exit 2
fi
if rustc_path=$(command -v rustc 2>/dev/null); then
    rustc_bin=$rustc_path
else
    rustc_bin="$(dirname -- "$cargo_bin")/rustc"
fi
if [[ ! -x "$rustc_bin" ]]; then
    printf 'rustc is not executable: %s\n' "$rustc_bin" >&2
    exit 2
fi
weights=${GPU_WEIGHTS:-"$repo_root/runs/pre_cuda_random/weights.bin"}
games=${GPU_PERF_GAMES:-64}
iters=${GPU_PERF_ITERS:-64}
depth=${GPU_PERF_DEPTH:-2}
reps=${GPU_PERF_REPS:-2}
profile_games=${GPU_PROFILE_GAMES:-16}
gpu_device=${GPU_DEVICE:-0}
micro_live=${GPU_MICRO_LIVE:-64}
micro_seconds=${GPU_MICRO_SECONDS:-10}

gen_bin="$engine_dir/target/release/examples/gpu_gen_bench"
micro_bin="$engine_dir/target/release/examples/gpu_bench"
base_gen="$baseline_dir/gpu_gen_bench"
base_micro="$baseline_dir/gpu_bench"

mkdir -p "$out_dir"
log="$out_dir/run.log"

say() {
    printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$log"
}

run() {
    say "$*"
    "$@" 2>&1 | tee -a "$log"
}

record_machine() {
    say "record machine and source state"
    {
        printf 'utc: '; date -u --iso-8601=seconds
        printf 'host: '; hostname
        printf 'mode: %s\n' "$mode"
        printf 'repo: %s\n' "$repo_root"
        printf 'commit: '; git -C "$repo_root" rev-parse HEAD
        printf 'rust: '; "$rustc_bin" --version
        printf 'cargo: '; "$cargo_bin" --version
        printf 'kernel: '; uname -srvo
        printf '\n-- source status --\n'
        git -C "$repo_root" status --short
        printf '\n-- source diff stat --\n'
        git -C "$repo_root" diff --stat
        printf '\n-- cpu --\n'
        lscpu
        printf '\n-- gpu --\n'
        nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE,MEMORY
        printf '\n-- gpu processes --\n'
        nvidia-smi pmon -c 1
        printf '\n-- cuda --\n'
        nvcc --version
        if command -v nsys >/dev/null 2>&1; then
            printf '\n-- nsys --\n'
            nsys --version
        fi
    } >"$out_dir/machine.txt" 2>&1
}

build() {
    run "$cargo_bin" build \
        --manifest-path "$engine_dir/Cargo.toml" \
        --release --features gpu \
        --example gpu_gen_bench --example gpu_bench
    run "$cargo_bin" test \
        --manifest-path "$engine_dir/Cargo.toml" \
        --release --features gpu --lib --no-run
}

phase_gate() {
    run "$cargo_bin" test \
        --manifest-path "$engine_dir/Cargo.toml" \
        --release --features gpu --lib phase_oracle \
        -- --nocapture --test-threads=1
}

full_gate() {
    run "$cargo_bin" test \
        --manifest-path "$engine_dir/Cargo.toml" \
        --release --features gpu --lib 'gpu::tests::' \
        -- --nocapture --test-threads=1
}

require_weights() {
    if [[ ! -f "$weights" ]]; then
        printf 'trained weights do not exist: %s\n' "$weights" >&2
        printf 'set GPU_WEIGHTS to an export_weights.py flat dump\n' >&2
        exit 2
    fi
}

bench_one() {
    local label=$1
    local binary=$2
    local output="$out_dir/$label.txt"
    say "$label"
    require_weights
    env CUDA_VISIBLE_DEVICES="$gpu_device" GPU_ONLY=1 GPU_WEIGHTS="$weights" \
        "$binary" "$games" "$iters" "$depth" 2>&1 | tee "$output" | tee -a "$log"
}

micro_one() {
    local label=$1
    local binary=$2
    local kind=$3
    local output="$out_dir/$label.txt"
    say "$label"
    require_weights
    if [[ "$kind" == zero ]]; then
        env CUDA_VISIBLE_DEVICES="$gpu_device" GPU_WEIGHTS="$weights" \
            GPU_ZERO_WEIGHTS=1 GPU_BENCH_CHECK="${GPU_BENCH_CHECK:-4}" \
            "$binary" "$micro_live" "$micro_seconds" "$iters" \
            2>&1 | tee "$output" | tee -a "$log"
    else
        env -u GPU_ZERO_WEIGHTS -u GPU_BENCH_CHECK \
            CUDA_VISIBLE_DEVICES="$gpu_device" GPU_WEIGHTS="$weights" \
            "$binary" "$micro_live" "$micro_seconds" "$iters" \
            2>&1 | tee "$output" | tee -a "$log"
    fi
}

candidate_bench() {
    micro_one candidate-micro-zero "$micro_bin" zero
    micro_one candidate-micro-trained "$micro_bin" trained
    bench_one candidate-trained "$gen_bin"
}

compare() {
    if [[ ! -x "$base_gen" ]]; then
        printf 'missing reference binary: %s\n' "$base_gen" >&2
        printf 'run tools/gpu_iter.sh baseline on the retained build first\n' >&2
        exit 2
    fi
    local r
    for ((r = 1; r <= reps; r++)); do
        if ((r % 2 == 1)); then
            micro_one "r${r}-base-micro-zero" "$base_micro" zero
            micro_one "r${r}-cand-micro-zero" "$micro_bin" zero
            micro_one "r${r}-base-micro-trained" "$base_micro" trained
            micro_one "r${r}-cand-micro-trained" "$micro_bin" trained
            bench_one "r${r}-base-trained" "$base_gen"
            bench_one "r${r}-cand-trained" "$gen_bin"
        else
            micro_one "r${r}-cand-micro-zero" "$micro_bin" zero
            micro_one "r${r}-base-micro-zero" "$base_micro" zero
            micro_one "r${r}-cand-micro-trained" "$micro_bin" trained
            micro_one "r${r}-base-micro-trained" "$base_micro" trained
            bench_one "r${r}-cand-trained" "$gen_bin"
            bench_one "r${r}-base-trained" "$base_gen"
        fi
    done
    python3 - "$out_dir" <<'PY' | tee "$out_dir/summary.txt" | tee -a "$log"
import pathlib, re, statistics, sys

root = pathlib.Path(sys.argv[1])
groups = {}
for path in sorted(root.glob("r*-*.txt")):
    text = path.read_text()
    rate = re.search(r"^solves(?:/s|/sec)\s+([0-9.]+)", text, re.M)
    tally = re.search(r"^tallies\s+(.+)$", text, re.M)
    answer = re.search(r"^answer hash\s+([0-9a-f]+)", text, re.M)
    if not rate:
        raise SystemExit(f"cannot parse {path}")
    key = path.stem.split("-", 1)[1]
    groups.setdefault(key, []).append(float(rate.group(1)))
    if tally:
        groups.setdefault(key + "-tallies", []).append(tally.group(1))
    if answer:
        groups.setdefault(key + "-hashes", []).append(answer.group(1))

print("\n== interleaved comparison ==")
for key in ("base-micro-zero", "cand-micro-zero", "base-micro-trained",
            "cand-micro-trained", "base-trained", "cand-trained"):
    values = groups.get(key, [])
    if values:
        print(f"{key:14} mean {statistics.fmean(values):8.2f}  runs {values}")
for kind in ("micro-zero", "micro-trained", "trained"):
    base = statistics.fmean(groups[f"base-{kind}"])
    cand = statistics.fmean(groups[f"cand-{kind}"])
    print(f"{kind:14} candidate/base {cand / base:8.4f}  delta {(cand / base - 1) * 100:+7.2f}%")
base_t = groups["base-trained-tallies"]
cand_t = groups["cand-trained-tallies"]
print(f"{'trained':14} identical tallies: {base_t == cand_t}")
base_h = groups["base-micro-zero-hashes"]
cand_h = groups["cand-micro-zero-hashes"]
print(f"{'micro-zero':14} bit-identical answers: {base_h == cand_h}")
if base_h != cand_h:
    raise SystemExit("zero-network answer hash changed")
PY
}

profile() {
    require_weights
    if ! command -v nsys >/dev/null 2>&1; then
        printf 'nsys is not installed\n' >&2
        exit 2
    fi
    local report="$out_dir/timeline"
    say "Nsight Systems profile"
    nsys profile \
        --trace=cuda,cublas,osrt \
        --sample=none \
        --cpuctxsw=none \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --output="$report" \
        env CUDA_VISIBLE_DEVICES="$gpu_device" GPU_ONLY=1 GPU_WEIGHTS="$weights" \
        "$gen_bin" "$profile_games" "$iters" "$depth" \
        2>&1 | tee "$out_dir/profile-program.txt" | tee -a "$log"
    nsys stats --force-export=true \
        --report=cuda_api_gpu_sum,cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,osrt_sum \
        "$report.nsys-rep" >"$out_dir/nsys-stats.txt"
    python3 "$repo_root/tools/nsys_summary.py" "$report.sqlite" \
        >"$out_dir/nsys-summary.txt"
    say "Nsight text reports: $out_dir/nsys-stats.txt"
}

micro() {
    micro_one candidate-micro-trained "$micro_bin" trained
}

freeze_baseline() {
    mkdir -p "$baseline_dir"
    cp -- "$gen_bin" "$base_gen"
    cp -- "$micro_bin" "$base_micro"
    {
        printf 'created: '; date -u --iso-8601=seconds
        printf 'commit: '; git -C "$repo_root" rev-parse HEAD
        git -C "$repo_root" status --short
        sha256sum "$base_gen" "$base_micro"
    } >"$baseline_dir/MANIFEST.txt"
    say "reference binaries frozen in $baseline_dir"
}

record_machine
case "$mode" in
    build)
        build
        ;;
    gate)
        build
        phase_gate
        full_gate
        ;;
    baseline)
        build
        phase_gate
        full_gate
        candidate_bench
        freeze_baseline
        ;;
    quick)
        build
        phase_gate
        candidate_bench
        ;;
    compare)
        build
        phase_gate
        compare
        ;;
    profile)
        build
        profile
        ;;
    micro)
        build
        micro
        ;;
    all)
        build
        phase_gate
        full_gate
        compare
        micro
        profile
        ;;
    *)
        printf 'unknown mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac

say "complete: $out_dir"
