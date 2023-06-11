from functools import partial
from operator import methodcaller

import torch
import torch.nn.functional as F
from torch import nn

from fakeglyph.utils.typing import copy_signature


class View(nn.Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.forward = methodcaller("view", *shape)


class ConvBNReLU2d(nn.Sequential):
    @copy_signature(nn.Conv2d.__init__)
    def __init__(self, *args, **kwargs) -> None:
        conv = nn.Conv2d(*args, **kwargs)
        bn = nn.BatchNorm2d(conv.out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)

    @classmethod
    def downsample(cls, in_channels: int):
        return cls(in_channels, in_channels * 2, 4, 2, 1, bias=False)


class ConvTransposeBNReLU2d(nn.Sequential):
    @copy_signature(nn.ConvTranspose2d.__init__)
    def __init__(self, *args, **kwargs) -> None:
        conv = nn.ConvTranspose2d(*args, **kwargs)
        bn = nn.BatchNorm2d(conv.out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)

    @classmethod
    def upsample(cls, in_channels: int):
        return cls(in_channels, in_channels // 2, 4, 2, 1, bias=False)


class ConvBNReLUEnd2d(ConvBNReLU2d):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels, num_channels, 3, 1, 1, bias=False)


class ResBlock2d(nn.Module):
    def __init__(self, num_channels: int, shrink_factor: int = 2) -> None:
        super().__init__()
        mid_channels = num_channels // shrink_factor
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        z = self.relu(x + y)
        return z


class Interpolate(nn.Module):
    @copy_signature(F.interpolate)  # Dirty hack: self shouldn't coincide with input
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.forward = partial(F.interpolate, *args, **kwargs)


class ResBlockInterpolate2d(nn.Sequential):
    def __init__(self, scale_factor: float, in_channels: int) -> None:
        out_channels = int(in_channels / scale_factor)
        super().__init__(
            ResBlock2d(in_channels),
            Interpolate(scale_factor=scale_factor, mode="bilinear"),
            ConvBNReLU2d(in_channels, out_channels, 1, bias=False),
        )

    @classmethod
    def downsample(cls, in_channels: int):
        return cls(0.5, in_channels)

    @classmethod
    def upsample(cls, in_channels: int):
        return cls(2.0, in_channels)
