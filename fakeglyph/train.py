from pathlib import Path
from typing import Callable, Iterator

import torch
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from .model.generative import GenerativeModel
from .utils.seed import set_seed


def train(
    seed: int | None,
    device: torch.device,
    batch_size: int,
    dataloader: DataLoader,
    model: GenerativeModel,
    num_epochs: int,
    optimizer_partial: Callable[[Iterator[Parameter]], Optimizer],
    lr_scheduler_partial: Callable[[Optimizer], LRScheduler],
    result_dir: Path,
) -> None:
    if seed is not None:
        set_seed(seed)

    model.to(device)
    optimizer = optimizer_partial(model.parameters())
    lr_scheduler = lr_scheduler_partial(optimizer)

    generator = model.generator
    fixed_noise = generator.generate_noise(batch_size)
    ckpt_path = result_dir / f"{type(model).__name__.lower()}.pt"
    sample_dir = result_dir / "samples"
    sample_dir.mkdir()
    writer = SummaryWriter(result_dir / "tensorboard")

    step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"epoch[{epoch}/{num_epochs}]")
        for data in pbar:
            x = data.to(fixed_noise)
            losses = model.step(x)
            optimizer.step()
            postfix = {name: f"{loss:.4f}" for name, loss in losses.items()}
            pbar.set_postfix(postfix)
            writer.add_scalars("losses", losses, step)
            step += 1

        torch.save(model.state_dict(), ckpt_path)

        model.eval()
        with torch.inference_mode():
            samples = generator(fixed_noise)
            sample_path = sample_dir / f"ep{epoch}.png"
            save_image(samples, sample_path)
            writer.add_images("samples", samples, epoch)

        lr_scheduler.step()
