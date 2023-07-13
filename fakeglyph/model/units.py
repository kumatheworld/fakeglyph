from collections.abc import Callable
from functools import partial
from operator import methodcaller

import torch
import torch.nn.functional as F
from torch import nn

from fakeglyph.utils.typehints import copy_signature


class Functional(nn.Module):
    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.forward = func

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.forward})"

    @classmethod
    def view(cls, *shape: int):
        return cls(methodcaller("view", *shape))

    @classmethod
    @copy_signature(F.interpolate)  # Dirty hack: cls shouldn't coincide with input
    def interpolate(cls, *args, **kwargs):
        return cls(partial(F.interpolate, *args, **kwargs))


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


class ResBlockInterpolate2d(nn.Sequential):
    def __init__(self, scale_factor: float, in_channels: int) -> None:
        out_channels = int(in_channels / scale_factor)
        super().__init__(
            ResBlock2d(in_channels),
            Functional.interpolate(scale_factor=scale_factor, mode="bilinear"),
            ConvBNReLU2d(in_channels, out_channels, 1, bias=False),
        )

    @classmethod
    def downsample(cls, in_channels: int):
        return cls(0.5, in_channels)

    @classmethod
    def upsample(cls, in_channels: int):
        return cls(2.0, in_channels)
