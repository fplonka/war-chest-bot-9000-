#!/usr/bin/env python3
"""Watch vast.ai for a box at least as good as ours and rent the first one.

    tools/vast.py                 # print matching offers, cheapest first, every 20 s
    tools/vast.py --rent          # rent the first match, then stop

The `vastai` CLI (pip install vastai; vastai set api-key ...) does the HTTP.
Good offers go within minutes, so this polls; the CLI's half-second start-up
is nothing against the 20 s cadence.
"""
import argparse
import json
import subprocess
import sys
import time

# The floor: two 24 GB cards, a modern many-core CPU, real PCIe bandwidth.
QUERY = ("num_gpus=2 gpu_ram>=24 total_flops>=68 cpu_cores_effective>=48 cpu_ram>=64 "
         "disk_space>=90 pcie_bw>=9 inet_down>=200 reliability>=0.98 cuda_max_good>=12.8 "
         "rentable=true verified=true")
IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"


def offers(max_dph):
    out = subprocess.run(["vastai", "search", "offers", f"{QUERY} dph<={max_dph}",
                          "-o", "dph", "--raw"], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else []


def line(o):
    return (f"{o['id']:>10} {o['gpu_name']:>10} x{o['num_gpus']} "
            f"{o['cpu_name'].strip()[:24]:24} {int(o['cpu_cores_effective']):3d}c "
            f"{int(o['cpu_ram']) // 1024:4d}G ram {int(o['disk_space']):5d}G disk "
            f"pcie {o['pcie_bw']:5.1f} rel {o['reliability2']:.3f} "
            f"${o['dph_total']:.3f}/h {o.get('geolocation', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-dph", type=float, default=0.30, help="price ceiling, $/h")
    ap.add_argument("--rent", action="store_true")
    ap.add_argument("--disk", type=int, default=100)
    args = ap.parse_args()
    while True:
        found = offers(args.max_dph)
        print(time.strftime("%H:%M:%S"), f"{len(found)} offers", flush=True)
        for o in found:
            print(line(o), flush=True)
        if found and args.rent:
            best = found[0]
            r = subprocess.run(["vastai", "create", "instance", str(best["id"]),
                                "--image", IMAGE, "--disk", str(args.disk),
                                "--ssh", "--direct", "--label", "warchest", "--raw"],
                               capture_output=True, text=True)
            print(r.stdout.strip() or r.stderr.strip())
            return 0 if r.returncode == 0 else 1
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
