from logging import getLogger

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from tqdm import tqdm

logger = getLogger(__name__)


class BinaryLetterDataset(Dataset):
    def __init__(
        self,
        file: torch.serialization.FILE_LIKE,
        font: ImageFont.FreeTypeFont,
        letters: str,
    ) -> None:
        try:
            images_nhw = torch.load(file)
        except FileNotFoundError:
            size = font.size
            image_tensors = []

            for text in tqdm(letters, "Converting letters to images"):
                with Image.new(mode="1", size=(size, size), color=False) as image:
                    draw = ImageDraw.Draw(image)
                    left, top, right, bottom = font.getbbox(text)
                    if left >= right or top >= bottom:
                        logger.warning(
                            f"Skipping '{text}' as bounding box "
                            f"{(left, top, right, bottom) = } looks invalid"
                        )
                        continue
                    x = (size - 1 - left - right) / 2
                    y = (size - 1 - top - bottom) / 2
                    draw.text((x, y), text, fill=True, font=font)
                    image_array = np.array(image)
                image_tensor = torch.from_numpy(image_array)
                image_tensors.append(image_tensor)

            images_nhw = torch.stack(image_tensors)
            torch.save(images_nhw, file)
            logger.info(f"Saving dataset at {file}")
        else:
            logger.info(f"Using cached dataset at {file}")

        images_n1hw = images_nhw.unsqueeze(1)
        self.images = images_n1hw
        logger.info(f"Loaded dataset of shape {tuple(images_n1hw.shape)}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]
