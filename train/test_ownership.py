import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import warchest
from value_net import CCOUNTS, Net


def inputs():
    rng = np.random.default_rng(17)
    x = torch.from_numpy(rng.standard_normal((2, warchest.PUBFEAT)).astype(np.float32))
    phi = torch.from_numpy(rng.standard_normal((8, CCOUNTS)).astype(np.float32))
    seg = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    weight = torch.full((8,), 0.5)
    return x, phi, weight, seg, 4


def main():
    torch.manual_seed(19)
    net = Net()
    args = inputs()
    value, pieces = net.evaluate(*args)
    target = torch.arange(warchest.N_LOCATIONS).remainder(3).unsqueeze(0).repeat(2, 1)
    loss = F.cross_entropy(net.ownership_logits(pieces[3], pieces[5]).flatten(0, 1),
                           target.flatten())
    loss.backward()
    for parameter in (net.ownership_out.weight, net.hex_stem.weight, net.cfg_g.weight):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad)

    before_value = value.detach().clone()
    before_flat = tuple(x.copy() for x in net.flat())
    with torch.no_grad():
        net.ownership_context.weight.normal_()
        net.ownership_out.weight.normal_()
    after_value = net(*args).detach()
    after_flat = net.flat()
    assert torch.equal(before_value, after_value)
    for before, after in zip(before_flat, after_flat):
        np.testing.assert_array_equal(before, after)
    assert "ownership_out.weight" in net.state_dict()
    print("ownership head test OK")


if __name__ == "__main__":
    main()
