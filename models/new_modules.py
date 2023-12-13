import torch
import torch.nn as nn


class RandomMask(nn.Module):
    """
    :param x: (N,C,H,W) tensor
    :return: (N,C,H,W) tensor
    """
    def __init__(self, p=0.5):
        super(RandomMask, self).__init__()
        self.p = p # p is the mask rate, instead of the preserving rate

    def forward(self, x):
        if self.training:
            N,C,H,W = x.size()
            mask = torch.rand(N, 1, H, W).to(x.device)
            mask = (mask - self.p).sign()
            mask = (mask + 1) / 2
            out = mask * x
        else: # eval
            out = x
        return out