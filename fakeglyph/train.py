from pathlib import Path
from typing import Callable, Iterator

import torch
from PIL.Image import Image
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
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
            sample_tensor = generator(fixed_noise).detach().cpu()
            sample_grid = make_grid(sample_tensor)
            sample: Image = to_pil_image(sample_grid)
            sample_name = f"ep{epoch}.png"
            sample_path = sample_dir / sample_name
            sample.save(sample_path)
            writer.add_images("samples", sample_tensor, epoch)

        lr_scheduler.step()
