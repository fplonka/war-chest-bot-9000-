"""Work-package-B verification on a CUDA box, via modal.

The laptop has no CUDA, so the oracle tests that need a device run here. The
container builds the engine with the `gpu` feature once and keeps the cargo
target directory in a volume, so a re-run only recompiles what changed.

    uvx modal run gpu_ci.py                          # every gpu test (T4)
    uvx modal run gpu_ci.py --test phase             # one test, by substring
    uvx modal run gpu_ci.py --no-release             # unoptimised backtraces
    uvx modal run gpu_ci.py --gpu L4 --bench "128 20 64"   # service benchmark
    uvx modal run gpu_ci.py --gpu L4 --gen "64 64 2"       # end-to-end benchmark

Optimised by default: the tests build a real subgame on the CPU first, and a
debug build spends minutes there before the first kernel runs.

GPU choice: T4 for correctness (cheapest), L4 for benchmarks (Ada, like the
target box), L40S for final estimates (nearly a 4090). Kernels are
NVRTC-compiled for the device at startup, so any of them works.
"""

import io
import os
import subprocess

import tarfile

import modal

REPO = os.path.dirname(os.path.abspath(__file__))
# Directories that are large and never needed to build or test.
SKIP = {"target", ".venv", "runs", ".git", "__pycache__", "papers", "node_modules"}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends "
        "curl build-essential pkg-config libssl-dev",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y "
        "--default-toolchain stable",
        # CARGO_HOME points at the cache volume, so the toolchain binaries
        # rustup installed under /root/.cargo must be reachable by path.
        "ln -sf /root/.cargo/bin/cargo /root/.cargo/bin/rustc /usr/local/bin/",
    )
)

app = modal.App("warchest-gpu-ci")
CARGO = modal.Volume.from_name("warchest-cargo", create_if_missing=True)
SRC = modal.Volume.from_name("warchest-src", create_if_missing=True)
ENV = {"CARGO_TARGET_DIR": "/root/warchest/target", "CARGO_HOME": "/root/warchest/cargo"}


def repo_tarball() -> io.BytesIO:
    """The repo, without the bulky build and runtime directories."""

    def keep(info):
        parts = info.name.split("/")[:-1]
        return None if any(p in SKIP for p in parts) else info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(REPO, arcname="repo", filter=keep)
    buf.seek(0)
    return buf


def _run(test: str, release: bool, bench: str, gen: str, timing: bool):
    os.makedirs("/repo", exist_ok=True)
    subprocess.run(
        ["tar", "xzf", "/root/src/repo.tgz", "-C", "/repo", "--strip-components=1"],
        check=True,
    )
    if bench or gen:
        example = "gpu_bench" if bench else "gpu_gen_bench"
        cmd = ["cargo", "run", "--release", "--features", "gpu",
               "--example", example, "--"] + (bench or gen).split()
        print("$", " ".join(cmd), flush=True)
        env = {**os.environ, **ENV}
        if timing:
            env["WARCHEST_GPU_TIMING"] = "1"
        p = subprocess.run(cmd, cwd="/repo/engine", env=env)
        if p.returncode != 0:
            raise SystemExit(p.returncode)
        return
    cmd = ["cargo", "test", "--features", "gpu", "--lib"]
    if release:
        cmd.append("--release")
    if test:
        cmd.append(test)
    cmd += ["--", "--nocapture", "--test-threads=1"]
    print("$", " ".join(cmd), flush=True)
    # Stream the output: a failing kernel can take a while to surface, and a
    # silent container for ten minutes is indistinguishable from a hang.
    p = subprocess.run(cmd, cwd="/repo/engine", env={**os.environ, **ENV})
    if p.returncode != 0:
        raise SystemExit(p.returncode)


# One function per GPU type; modal fixes the GPU at declaration.
common = dict(image=image, timeout=1800,
              volumes={"/root/warchest": CARGO, "/root/src": SRC})


@app.function(gpu="T4", **common)
def run_t4(test: str, release: bool, bench: str, gen: str, timing: bool):
    _run(test, release, bench, gen, timing)


@app.function(gpu="L4", **common)
def run_l4(test: str, release: bool, bench: str, gen: str, timing: bool):
    _run(test, release, bench, gen, timing)


@app.function(gpu="L40S", **common)
def run_l40s(test: str, release: bool, bench: str, gen: str, timing: bool):
    _run(test, release, bench, gen, timing)


@app.local_entrypoint()
def main(test: str = "", no_release: bool = False, bench: str = "",
         gen: str = "", timing: bool = False, gpu: str = "T4"):
    with SRC.batch_upload(force=True) as batch:
        batch.put_file(repo_tarball(), "/repo.tgz")
    fn = {"T4": run_t4, "L4": run_l4, "L40S": run_l40s}[gpu.upper()]
    fn.remote(test, not no_release, bench, gen, timing)
