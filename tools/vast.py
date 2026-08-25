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
import http.client
from pathlib import Path

API = "/api/v0"
HOST = "console.vast.ai"
KEY = Path("~/.config/vastai/vast_api_key").expanduser().read_text().strip()
IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"
# The floor: two 24 GB cards, a modern many-core CPU, real PCIe bandwidth.
FLOOR = {"num_gpus": {"eq": 2}, "gpu_ram": {"gte": 24000}, "total_flops": {"gte": 68},
         "cpu_cores_effective": {"gte": 48}, "cpu_ram": {"gte": 64000},
         "disk_space": {"gte": 90}, "pcie_bw": {"gte": 9}, "inet_down": {"gte": 200},
         "reliability2": {"gte": 0.98}, "cuda_max_good": {"gte": 12.8},
         "verified": {"eq": True}, "external": {"eq": False},
         "rentable": {"eq": True}, "rented": {"eq": False}}


conn = None


def call(method, path, body):
    """One kept-alive TLS connection: the handshake, not the server, was most
    of a poll's latency."""
    global conn
    if conn is None:
        conn = http.client.HTTPSConnection(HOST, timeout=20)
    try:
        conn.request(method, f"{API}{path}", body=json.dumps(body),
                     headers={"Authorization": f"Bearer {KEY}",
                              "Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
    except (http.client.HTTPException, OSError):
        conn.close()
        conn = None
        raise
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data[:120].decode(errors='replace')}")
    return json.loads(data)


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
    seen, polls, delay = set(), 0, 1.0
    while True:
        # The API allows five requests per five seconds ("limit":5.0 in its
        # 429 body), so one poll a second is the ceiling; back off on a 429.
        time.sleep(delay)
        try:
            found = offers(args.max_dph, args.disk)
        except Exception as e:
            delay = min(60.0, delay * 2)
            print(time.strftime("%H:%M:%S"), f"poll failed: {e}; delay {delay:.1f}s", flush=True)
            continue
        delay = max(1.0, delay * 0.9)
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
        if polls % 3600 == 0:
            print(time.strftime("%H:%M:%S"), f"{polls} polls, delay {delay:.1f}s, still watching", flush=True)


if __name__ == "__main__":
    sys.exit(main())
