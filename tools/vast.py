#!/usr/bin/env python3
"""Watch vast.ai for a box at least as good as ours and rent the first one.

    tools/vast.py                 # print each new matching offer, cheapest first
    tools/vast.py --rent          # rent the first match, then stop

Good offers are gone within seconds, so this talks to the API directly and
polls as fast as it answers. The key is the one `vastai set api-key` saved.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://console.vast.ai/api/v0"
KEY = Path("~/.config/vastai/vast_api_key").expanduser().read_text().strip()
IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"
# The floor: two 24 GB cards, a modern many-core CPU, real PCIe bandwidth.
FLOOR = {"num_gpus": {"eq": 2}, "gpu_ram": {"gte": 24000}, "total_flops": {"gte": 68},
         "cpu_cores_effective": {"gte": 48}, "cpu_ram": {"gte": 64000},
         "disk_space": {"gte": 90}, "pcie_bw": {"gte": 9}, "inet_down": {"gte": 200},
         "reliability2": {"gte": 0.98}, "cuda_max_good": {"gte": 12.8},
         "verified": {"eq": True}, "external": {"eq": False},
         "rentable": {"eq": True}, "rented": {"eq": False}}


def call(method, path, body):
    req = urllib.request.Request(f"{API}{path}", method=method,
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def offers(max_dph, disk):
    q = dict(FLOOR, dph_total={"lte": max_dph}, order=[["dph_total", "asc"]],
             type="on-demand", allocated_storage=disk)
    return call("POST", "/bundles/", q)["offers"]


def rent(offer, disk):
    return call("PUT", f"/asks/{offer['id']}/", {
        "client_id": "me", "image": IMAGE, "disk": disk, "label": "warchest",
        "runtype": "ssh_direc ssh_proxy", "env": {}})


def line(o):
    return (f"{o['id']:>10} {o['gpu_name']:>10} x{o['num_gpus']} "
            f"{o['cpu_name'].strip()[:24]:24} {int(o['cpu_cores_effective']):3d}c "
            f"{int(o['cpu_ram']) // 1024:4d}G ram {int(o['disk_space']):5d}G disk "
            f"pcie {o['pcie_bw']:5.1f} rel {o['reliability2']:.3f} "
            f"${o['dph_total']:.3f}/h {o.get('geolocation', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-dph", type=float, default=0.27, help="price ceiling, $/h with disk")
    ap.add_argument("--rent", action="store_true")
    ap.add_argument("--disk", type=int, default=80)
    args = ap.parse_args()
    seen, polls = set(), 0
    while True:
        try:
            found = offers(args.max_dph, args.disk)
        except Exception as e:  # a 429 or a blip is not a reason to stop watching
            print(time.strftime("%H:%M:%S"), f"poll failed: {e}", flush=True)
            time.sleep(2)
            continue
        polls += 1
        for o in found:
            if o["id"] not in seen:
                seen.add(o["id"])
                print(time.strftime("%H:%M:%S"), line(o), flush=True)
        if found and args.rent:
            try:
                print("RENTED", json.dumps(rent(found[0], args.disk)), line(found[0]), flush=True)
                return 0
            except Exception as e:  # taken under us: keep polling
                print(time.strftime("%H:%M:%S"), f"rent failed: {e}", flush=True)
        if polls % 600 == 0:
            print(time.strftime("%H:%M:%S"), f"{polls} polls, still watching", flush=True)


if __name__ == "__main__":
    sys.exit(main())
