import torch
import numpy as np


__all__ = [
    "transform_img_from_dataloader",
    "transform_dataloader_from_img",
]


def transform_img_from_dataloader(img: torch.Tensor):
    """
    Transformation to interpret images read by dataloader
    """
    img = img.numpy().copy().T.astype(np.uint8)
    img = np.rot90(img, 1)
    img = np.flipud(img)
    return img

def transform_dataloader_from_img(img: np.ndarray):
    img = np.flipud(img)
    img = np.rot90(img, -1)
    img = torch.from_numpy(img.T).to(torch.uint8)
    return img