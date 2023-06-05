from abc import ABC, abstractmethod
from operator import methodcaller
from typing import Literal

import torch
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy, softplus

from ..utils.typing import T2T, Losses, T2TModule
from .generator import Generator

softplus: T2T


class GenerativeModel(ABC, Module):
    def __init__(self, generator: Generator) -> None:
        super().__init__()
        self.generator = generator

    @abstractmethod
    def step(self, x: torch.Tensor) -> Losses:
        ...


class VAE(GenerativeModel):
    def __init__(
        self,
        encoder: T2TModule,
        decoder: Generator,
        beta: float,
        reduction: Literal["batchmean", "mean"],
    ) -> None:
        super().__init__(decoder)
        self.encoder = encoder

        self.beta = beta
        self.reducer: T2T
        match reduction:
            case "batchmean":
                self.reducer = lambda t: t.sum() / t.shape[0]
            case "mean":
                self.reducer = methodcaller("mean")
            case _:
                raise ValueError(f"{reduction} is not a valid value for reduction")

    def step(self, x: torch.Tensor) -> Losses:
        self.zero_grad()
        encoder = self.encoder
        decoder = self.generator

        y = encoder(x)
        mean, logstd = y.chunk(chunks=2, dim=1)
        kl_div_elems = (mean.square() + torch.expm1(2 * logstd)) / 2 - logstd
        kl_div = self.reducer(kl_div_elems)

        noise = decoder.generate_noise(x.shape[0])
        std = logstd.exp()
        z = mean + std * noise
        w = decoder(z)
        loss_rec_elem = binary_cross_entropy(w, x, reduction="none")
        loss_rec = self.reducer(loss_rec_elem)

        loss = loss_rec + self.beta * kl_div
        loss.backward()

        losses = {"kl_div": kl_div.item(), "loss_rec": loss_rec.item()}
        return losses


class GAN(GenerativeModel):
    def __init__(self, generator: Generator, discriminator: T2TModule) -> None:
        super().__init__(generator)
        self.discriminator = discriminator

    def step(self, x: torch.Tensor) -> Losses:
        generator = self.generator
        discriminator = self.discriminator

        discriminator.zero_grad()
        logits_dr = discriminator(x)
        loss_dr = softplus(-logits_dr).mean()

        fake = generator.sample(x.shape[0])
        logits_df = discriminator(fake.detach())
        loss_df = softplus(logits_df).mean()
        loss_d = loss_dr + loss_df
        loss_d.backward()

        generator.zero_grad()
        logits_dgf = discriminator(fake)
        loss_g = softplus(-logits_dgf).mean()
        loss_g.backward()

        losses = {
            "loss_dr": loss_dr.item(),
            "loss_df": loss_df.item(),
            "loss_g": loss_g.item(),
        }
        return losses
