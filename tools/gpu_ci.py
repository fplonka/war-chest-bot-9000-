"""Work-package-B verification on a CUDA box (modal).

Bare minimum GPU time: the container builds the engine with the `gpu`
feature once (cargo target cached in a volume) and runs the oracle tests
that need CUDA. The CPU-only tests and the torch spec run on the laptop.

    uvx modal run tools/gpu_ci.py            # the default: unit tests
    uvx modal run tools/gpu_ci.py --bench    # the B5.6 benchmark
    uvx modal run tools/gpu_ci.py --ladder   # the B5.5 same-weights ladder

Uses the cheapest modal GPU that works (T4; the kernels compile to
compute_75, which T4 is).
"""

import modal
from modal.mount import Mount

VOL = modal.Volume.from_name("warchest-cargo", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends curl build-essential pkg-config libssl-dev",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    )
    .pip_install("numpy")
)

app = modal.App("warchest-gpu-ci")

REPO = Mount()
REPO.add_local_dir(
    "/Users/filip/Code/warchest-cuda", remote_path="/repo"
)

ENV = {"CARGO_TARGET_DIR": "/root/repo-target", "CARGO_HOME": "/root/.cargo"}


@app.function(
    image=image, gpu="T4", timeout=3600, mounts=[REPO],
    volumes={"/root/.cargo": VOL, "/root/repo-target": VOL},
    secrets=[],
)
def run_tests(extra: str = ""):
    import os
    import subprocess

    os.chdir("/repo/engine")
    cmd = ["cargo", "test", "--features", "gpu", "--lib"] + extra.split()
    r = subprocess.run(cmd, env={**os.environ, **ENV},
                       capture_output=True, text=True)
    print(r.stdout[-12000:])
    print(r.stderr[-4000:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)


@app.function(
    image=image, gpu="T4", timeout=3600, mounts=[REPO],
    volumes={"/root/.cargo": VOL, "/root/repo-target": VOL},
)
def bench():
    import os
    import subprocess

    os.chdir("/repo/engine")
    cmd = ["cargo", "run", "--release", "--features", "gpu", "--example",
           "gpu_bench"]
    r = subprocess.run(cmd, env={**os.environ, **ENV},
                       capture_output=True, text=True)
    print(r.stdout[-12000:])
    print(r.stderr[-4000:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)


@app.function(
    image=image, gpu="T4", timeout=3600, mounts=[REPO],
    volumes={"/root/.cargo": VOL, "/root/repo-target": VOL},
)
def ladder():
    import os
    import subprocess

    os.chdir("/repo/engine")
    cmd = ["cargo", "run", "--release", "--features", "gpu", "--example",
           "gpu_ladder"]
    r = subprocess.run(cmd, env={**os.environ, **ENV},
                       capture_output=True, text=True)
    print(r.stdout[-12000:])
    print(r.stderr[-4000:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    import sys

    with app.run():
        if "--bench" in sys.argv:
            bench.remote()
        elif "--ladder" in sys.argv:
            ladder.remote()
        else:
            extra = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
            run_tests.remote(extra)
