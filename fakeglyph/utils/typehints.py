from typing import Callable, TypeAlias, TypeVar

import torch
from torch.nn import Module

F = TypeVar("F", bound=Callable)


def copy_signature(_: F) -> Callable[..., F]:
    return lambda f: f


T2T: TypeAlias = Callable[[torch.Tensor], torch.Tensor]


class T2TModule(Module):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


Losses: TypeAlias = dict[str, float]
