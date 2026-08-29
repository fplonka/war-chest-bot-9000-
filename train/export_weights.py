
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from value_net import Net


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net()
    net.load_state_dict(ck["value"])
    return net


def write_bin(net, path):
    with open(path, "wb") as f:
        for a in net.flat():
            a = np.ascontiguousarray(a, np.float32)
            f.write(struct.pack("<I", a.size))
            f.write(a.tobytes())


def main():
    src, dst = sys.argv[1], sys.argv[2]
    write_bin(load(src), dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
