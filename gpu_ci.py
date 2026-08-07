"""Work-package-B verification on a CUDA box (modal).

Bare minimum GPU time: the container builds the engine with the `gpu`
feature once (cargo target cached in a volume) and runs the oracle tests
that need CUDA. The CPU-only tests and the torch spec run on the laptop.

    uvx modal run gpu_ci.py            # the default: unit tests
    uvx modal run gpu_ci.py --bench    # the B5.6 benchmark
    uvx modal run gpu_ci.py --ladder   # the B5.5 same-weights ladder

Uses the cheapest modal GPU that works (T4; the kernels compile to
compute_75, which T4 is). Edit FILTER to run one test.
"""

import io
import os
import subprocess
import sys
import tarfile

import modal
from modal.mount import Mount

# cargo test filter ("" = all; e.g. "phase_oracle", "full_solve_oracle")
FILTER = "phase_oracle"

VOL = modal.Volume.from_name("warchest-cargo", create_if_missing=True)
SRC = modal.Volume.from_name("warchest-src", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends curl build-essential pkg-config libssl-dev gdb",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        "ln -sf /root/.cargo/bin/cargo /usr/local/bin/cargo && ln -sf /root/.cargo/bin/rustc /usr/local/bin/rustc",
    )
    .pip_install("numpy")
)

app = modal.App("warchest-gpu-ci")

ENV = {
    "CARGO_TARGET_DIR": "/root/warchest/target",
    "CARGO_HOME": "/root/warchest/cargo",
}


def repo_tarball() -> io.BytesIO:
    """The repo, without the bulky build/runtime dirs."""
    skip = {"target", ".venv", "runs", ".git", "__pycache__", "papers", "node_modules"}

    def filt(i):
        rel = i.name.split("/repo/")[-1] if "/repo/" in i.name else i.name
        return None if any(c in skip for c in rel.split("/")[:-1]) else i

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("/Users/filip/Code/warchest-cuda", arcname="repo", filter=filt)
    buf.seek(0)
    return buf


@app.function(image=image, gpu="T4", timeout=3600,
              volumes={"/root/warchest": VOL, "/root/src": SRC})
def run_tests():
    os.makedirs("/repo", exist_ok=True)
    subprocess.run(["tar", "xzf", "/root/src/repo.tgz", "-C", "/repo", "--strip-components=1"], check=True)
    os.chdir("/repo/engine")
    cmd = ["cargo", "test", "--features", "gpu", "--lib"]
    if FILTER:
        cmd.append(FILTER)
    cmd += ["--", "--nocapture", "--test-threads=1"]
    r = subprocess.run(cmd, env={**os.environ, **ENV},
                       capture_output=True, text=True)
    print(r.stdout[-20000:])
    print(r.stderr[-4000:])
    if r.returncode != 0:
        # always rerun under gdb for a backtrace (fast: the binary is built)
        import glob
        bins = glob.glob("/root/warchest/target/debug/deps/warchest-*")
        bins = [b for b in bins if "d" in b and b.endswith(".d") is False]
        # the lib test binary is the one with a long hash, no extension
        bins = [b for b in bins if "." not in b.split("/")[-1]]
        if bins:
            g = subprocess.run(["gdb", "-batch", "-ex", "run", "-ex", "bt", bins[0], "phase_oracle"],
                               env={**os.environ, **ENV}, capture_output=True, text=True)
            print("GDB OUTPUT:\n", g.stdout[-15000:], g.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)


@app.function(image=image, gpu="T4", timeout=3600,
              volumes={"/root/warchest": VOL, "/root/src": SRC})
def bench():
    os.makedirs("/repo", exist_ok=True)
    subprocess.run(["tar", "xzf", "/root/src/repo.tgz", "-C", "/repo", "--strip-components=1"], check=True)
    os.chdir("/repo/engine")
    cmd = ["cargo", "run", "--release", "--features", "gpu", "--example", "gpu_bench"]
    r = subprocess.run(cmd, env={**os.environ, **ENV},
                       capture_output=True, text=True)
    print(r.stdout[-20000:])
    print(r.stderr[-4000:])
    if r.returncode != 0:
        # always rerun under gdb for a backtrace (fast: the binary is built)
        import glob
        bins = glob.glob("/root/warchest/target/debug/deps/warchest-*")
        bins = [b for b in bins if "d" in b and b.endswith(".d") is False]
        # the lib test binary is the one with a long hash, no extension
        bins = [b for b in bins if "." not in b.split("/")[-1]]
        if bins:
            g = subprocess.run(["gdb", "-batch", "-ex", "run", "-ex", "bt", bins[0], "phase_oracle"],
                               env={**os.environ, **ENV}, capture_output=True, text=True)
            print("GDB OUTPUT:\n", g.stdout[-15000:], g.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)


@app.function(image=image, gpu="T4", timeout=3600,
              volumes={"/root/warchest": VOL, "/root/src": SRC})
def ladder():
    os.makedirs("/repo", exist_ok=True)
    subprocess.run(["tar", "xzf", "/root/src/repo.tgz", "-C", "/repo", "--strip-components=1"], check=True)
    os.chdir("/repo/engine")
    cmd = ["cargo", "run", "--release", "--features", "gpu", "--example", "gpu_ladder"]
    r = subprocess.run(cmd, env={**os.environ, **ENV},
                       capture_output=True, text=True)
    print(r.stdout[-20000:])
    print(r.stderr[-4000:])
    if r.returncode != 0:
        # always rerun under gdb for a backtrace (fast: the binary is built)
        import glob
        bins = glob.glob("/root/warchest/target/debug/deps/warchest-*")
        bins = [b for b in bins if "d" in b and b.endswith(".d") is False]
        # the lib test binary is the one with a long hash, no extension
        bins = [b for b in bins if "." not in b.split("/")[-1]]
        if bins:
            g = subprocess.run(["gdb", "-batch", "-ex", "run", "-ex", "bt", bins[0], "phase_oracle"],
                               env={**os.environ, **ENV}, capture_output=True, text=True)
            print("GDB OUTPUT:\n", g.stdout[-15000:], g.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)


@app.local_entrypoint()
def main():
    print("uploading repo tarball...")
    with SRC.batch_upload(force=True) as batch:
        batch.put_file(repo_tarball(), "/repo.tgz")
    if "--bench" in sys.argv:
        bench.remote()
    elif "--ladder" in sys.argv:
        ladder.remote()
    else:
        run_tests.remote()
