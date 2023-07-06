from collections.abc import Sequence
from functools import cached_property

import torch

from fakeglyph.utils.typehints import T2TModule


class Generator(T2TModule):
    def __init__(self, input_shape: Sequence[int], layers: T2TModule) -> None:
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.layers = layers

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)

    @cached_property
    def _empty(self) -> torch.Tensor:
        """
        A workaround to generate noise of the same dtype and device as self.
        Caches the first parameters' information.
        Expected to be called after to().
        """
        first_params = next(self.parameters())
        empty = torch.empty(()).to(first_params)
        return empty

    def generate_noise(self, batch_size: int) -> torch.Tensor:
        empty = self._empty
        z = torch.randn(
            batch_size,
            *self.input_shape,
            dtype=empty.dtype,
            layout=empty.layout,
            device=empty.device
        )
        return z

    def sample(self, num_samples: int) -> torch.Tensor:
        z = self.generate_noise(num_samples)
        y = self(z)
        return y
