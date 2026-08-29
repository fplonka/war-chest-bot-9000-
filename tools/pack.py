import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def selected(snapshots):
    if not snapshots:
        return []
    end = snapshots[-1]["t"]
    targets = [0.0, end / 8, end / 4, end / 2, end]
    indices = {min(range(len(snapshots)),
                   key=lambda i: abs(snapshots[i]["t"] - target))
               for target in targets}
    indices.add(len(snapshots) - 1)
    return [snapshots[i] for i in sorted(indices)]


def pack(run, binary, out_dir=None, snapshot=None, name=None):
    import sys

    sys.path.insert(0, str(ROOT / "train"))
    import torch
    from export_weights import write_bin
    from value_net import Net

    binary = Path(binary)
    if not binary.exists():
        raise SystemExit(f"{binary} does not exist. Build it with "
                         "cargo build --release --features gpu --bin bot")
    run = Path(run)
    out_dir = Path(out_dir) if out_dir else run / "bots"
    log = json.loads((run / "log.json").read_text())
    cfg = log.get("cfg", {})
    snapshots = log.get("snapshots", [])
    if snapshot is None:
        snaps = selected(snapshots)
    else:
        snaps = [snap for snap in snapshots if snap["label"] == snapshot]
    if not snaps:
        raise SystemExit(f"{run} has no snapshot {snapshot!r}")
    if name and len(snaps) > 1:
        raise SystemExit("--name needs a single --snapshot")
    if out_dir == run / "bots" and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        final = snap is snapshots[-1]
        stem = Path(snap["file"]).stem.removeprefix("snap_")
        bot = name or ("final" if final else f"{stem}-{snap['label']}")
        directory = out_dir / bot
        directory.mkdir(parents=True, exist_ok=False)
        write_bin(net, directory / "weights.bin")
        shutil.copy2(binary, directory / "bot")
        (directory / "bot.json").write_text(json.dumps({
            "format": 1,
            "name": f"{run.name}.{bot}",
            "sha": checkpoint.get("git", cfg.get("git", "")),
            "binary": digest(directory / "bot"),
            "mind": "sog",
            "weights": "weights.bin",
            "search": search,
            "minutes": round(snap["t"] / 60.0, 3),
            "note": f"{run.name} {snap['label']}, {snap['t'] / 60:.0f} min",
        }, indent=1) + "\n")
        made.append(directory)
        print(f"packed {directory}")
    return made


def main():
    ap = argparse.ArgumentParser(description="Pack selected run checkpoints as bots.")
    ap.add_argument("run")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--bin", default=str(ROOT / "engine/target/release/bot"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pack(args.run, Path(args.bin).resolve(), args.out, args.snapshot, args.name)


if __name__ == "__main__":
    main()
