import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOT = ROOT / "engine/target/release/bot"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def pack(run):
    import sys

    sys.path.insert(0, str(ROOT / "train"))
    import torch
    from export_weights import write_bin
    from value_net import Net

    if not BOT.exists():
        raise SystemExit(f"{BOT} does not exist. Run tools/box.sh go")
    run = Path(run)
    out_dir = run / "bots"
    log = json.loads((run / "log.json").read_text())
    cfg = log.get("cfg", {})
    snapshots = log.get("snapshots", [])
    snaps = selected(snapshots)
    if not snaps:
        raise SystemExit(f"{run} has no snapshots")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for snap in snaps:
        checkpoint = torch.load(run / snap["file"], map_location="cpu",
                                weights_only=False)
        net = Net()
        net.load_state_dict(checkpoint["value"])
        search = checkpoint.get("search")
        required = ["s", "c", "batch", "rounds", "cfr", "puct", "prior_temp"]
        missing = [key for key in required if not isinstance(search, dict) or key not in search]
        if missing:
            raise SystemExit(f"{snap['file']} is missing packed search fields: {', '.join(missing)}")
        search = {key: search[key] for key in required}
        final = snap is snapshots[-1]
        stem = Path(snap["file"]).stem.removeprefix("snap_")
        bot = "final" if final else f"{stem}-{snap['label']}"
        directory = out_dir / bot
        directory.mkdir(parents=True, exist_ok=False)
        write_bin(net, directory / "weights.bin")
        shutil.copy2(BOT, directory / "bot")
        (directory / "bot.json").write_text(json.dumps({
            "format": 2,
            "name": f"{run.name}.{bot}",
            "sha": checkpoint.get("git", cfg.get("git", "")),
            "binary": digest(directory / "bot"),
            "weights": "weights.bin",
            "weights_sha": digest(directory / "weights.bin"),
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
    args = ap.parse_args()
    pack(args.run)


if __name__ == "__main__":
    main()
