import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def pack(run, binary, out_dir, snapshot=None, name=None):
    import sys

    sys.path.insert(0, str(ROOT / "train"))
    import torch
    from export_weights import write_bin
    from value_net import Net

    if not Path(binary).exists():
        raise SystemExit(f"{binary} does not exist. Build it:\n"
                         f"  cd engine && cargo build --release --features gpu --bin ladder")

    run = Path(run)
    log = json.loads((run / "log.json").read_text())
    cfg = log.get("cfg", {})
    snaps = [s for s in log.get("snapshots", [])
             if snapshot is None or s["label"] == snapshot]
    if not snaps:
        raise SystemExit(f"{run} has no snapshot {snapshot!r}")
    if name and len(snaps) > 1:
        raise SystemExit("--name needs a single --snapshot")
    made = []
    for snap in snaps:
        checkpoint = torch.load(run / snap["file"], map_location="cpu",
                                weights_only=False)
        net = Net()
        net.load_state_dict(checkpoint["value"])
        search = {"s": cfg.get("s", 512),
                  "c": cfg.get("c", 8.0),
                  "batch": cfg.get("round_batch", 8),
                  "rounds": cfg.get("rounds", 0),
                  "puct": 1.5,
                  "prior_temp": 1.0,
                  "cfr": "dcfr"}
        bot = name or f"{run.name}.{snap['label']}"
        directory = out_dir / bot
        directory.mkdir(parents=True, exist_ok=True)
        write_bin(net, directory / "weights.bin")
        shutil.copy2(binary, directory / "bot")
        (directory / "bot.json").write_text(json.dumps({
            "name": bot,
            "sha": checkpoint.get("git", ""),
            "binary": digest(directory / "bot"),
            "mind": "sog",
            "weights": "weights.bin",
            "search": search,
            "minutes": round(snap["t"] / 60.0, 1),
            "note": f"{run.name} {snap['label']}, {snap['t'] / 60:.0f} min",
        }, indent=1) + "\n")
        made.append(directory)
        print(f"packed {directory}")
    return made


def main():
    ap = argparse.ArgumentParser(description="Archive run snapshots as bots.")
    ap.add_argument("run")
    ap.add_argument("--snapshot", default=None,
                    help="one snapshot label instead of all of them")
    ap.add_argument("--name", default=None, help="bot name (default run.label)")
    ap.add_argument("--bin", default=str(ROOT / "engine/target/release/ladder"),
                    help="the binary to record alongside the weights")
    ap.add_argument("--out", default="bots")
    args = ap.parse_args()
    pack(args.run, Path(args.bin).resolve(), Path(args.out), args.snapshot, args.name)


if __name__ == "__main__":
    main()
