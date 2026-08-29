
import io
import os
import subprocess

import tarfile

import modal

REPO = os.path.dirname(os.path.abspath(__file__))
SKIP = {"target", ".venv", "runs", ".git", "__pycache__", "papers", "node_modules"}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends "
        "curl build-essential pkg-config libssl-dev",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y "
        "--default-toolchain stable",
        "ln -sf /root/.cargo/bin/cargo /root/.cargo/bin/rustc /usr/local/bin/",
    )
)

app = modal.App("warchest-gpu-ci")
CARGO = modal.Volume.from_name("warchest-cargo", create_if_missing=True)
SRC = modal.Volume.from_name("warchest-src", create_if_missing=True)
ENV = {"CARGO_TARGET_DIR": "/root/warchest/target", "CARGO_HOME": "/root/warchest/cargo"}


def repo_tarball() -> io.BytesIO:

    def keep(info):
        parts = info.name.split("/")[:-1]
        return None if any(p in SKIP for p in parts) else info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(REPO, arcname="repo", filter=keep)
    buf.seek(0)
    return buf


def _run(test: str, release: bool):
    os.makedirs("/repo", exist_ok=True)
    subprocess.run(
        ["tar", "xzf", "/root/src/repo.tgz", "-C", "/repo", "--strip-components=1"],
        check=True,
    )
    cmd = ["cargo", "test", "--features", "gpu", "--lib", "--tests"]
    if release:
        cmd.append("--release")
    if test:
        cmd.append(test)
    cmd += ["--", "--nocapture", "--test-threads=1"]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd="/repo/engine", env={**os.environ, **ENV})
    if p.returncode != 0:
        raise SystemExit(p.returncode)


common = dict(image=image, timeout=1800,
              volumes={"/root/warchest": CARGO, "/root/src": SRC})


@app.function(gpu="T4", **common)
def run_t4(test: str, release: bool):
    _run(test, release)


@app.function(gpu="L4", **common)
def run_l4(test: str, release: bool):
    _run(test, release)


@app.function(gpu="L40S", **common)
def run_l40s(test: str, release: bool):
    _run(test, release)


@app.local_entrypoint()
def main(test: str = "", no_release: bool = False, gpu: str = "T4"):
    with SRC.batch_upload(force=True) as batch:
        batch.put_file(repo_tarball(), "/repo.tgz")
    fn = {"T4": run_t4, "L4": run_l4, "L40S": run_l40s}[gpu.upper()]
    fn.remote(test, not no_release)
