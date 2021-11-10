from typing import Union, List
from kkdetectron2.util.com import check_type_list
import torch
import numpy as np


__all__ = [
    "transform_img_from_dataloader",
    "transform_dataloader_from_img",
    "check_and_conv_bbox_data",
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "xywh_to_xyah",
    "xyah_to_xywh",
    "xyxy_to_xyah",
    "xyah_to_xyxy",
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

def check_and_conv_bbox_data(data: Union[np.ndarray, List[float], List[List[float]]]) -> np.ndarray:
    assert isinstance(data, np.ndarray) or isinstance(data, list)
    is_list = False
    if isinstance(data, list):
        is_list = True
        if isinstance(data[0], list):
            assert check_type_list(data, list, [int, float])
        else:
            assert check_type_list(data, [int, float])
        data = np.array(data)
    assert len(data.shape) in [1, 2]
    if len(data.shape) == 1:
        assert data.shape[0] == 4
    else:
        assert data.shape[1] == 4
    return data.copy(), is_list

def xyxy_to_xywh(data: Union[np.ndarray, List[float], List[List[float]]]):
    """
    xyxy: x1, y1, x2, y2
    xywh: (top left) x min, (top left) y min, width, height
    """
    data, is_list = check_and_conv_bbox_data(data)
    if len(data.shape) == 1:
        x1, y1, x2, y2 = data
        data = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        data = data.copy()
        data[:, 2] = data[:, 2] - data[:, 0]
        data[:, 3] = data[:, 3] - data[:, 1]
    if is_list: return data.tolist()
    return data

def xywh_to_xyxy(data: Union[np.ndarray, List[float], List[List[float]]]):
    data, is_list = check_and_conv_bbox_data(data)
    if len(data.shape) == 1:
        xmin, ymin, w, h = data
        data = np.array([xmin, ymin, xmin + w, ymin + h])
    else:
        data = data.copy()
        data[:, 2] = data[:, 0] + data[:, 2]
        data[:, 3] = data[:, 1] + data[:, 3]
    if is_list: return data.tolist()
    return data

def xywh_to_xyah(data: Union[np.ndarray, List[float], List[List[float]]]):
    """
    xywh: (top left) x min, (top left) y min, width, height
    xyah: center x, center y, aspect ratio (width / height), height
    """
    data, is_list = check_and_conv_bbox_data(data)
    if len(data.shape) == 1:
        xmin, ymin, w, h = data
        data = np.array([xmin + w/2., ymin + h/2., w/h, h])
    else:
        data = data.copy()
        data[:, 0] = data[:, 0] + (data[:, 2] / 2.)
        data[:, 1] = data[:, 1] + (data[:, 3] / 2.)
        data[:, 2] = data[:, 2] / data[:, 3]
    if is_list: return data.tolist()
    return data

def xyah_to_xywh(data: Union[np.ndarray, List[float], List[List[float]]]):
    data, is_list = check_and_conv_bbox_data(data)
    if len(data.shape) == 1:
        xcen, ycen, a, h = data
        w    = a * h
        data = np.array([xcen - w/2., ycen -h/2., w, h])
    else:
        data = data.copy()
        w    = (data[:, 2] * data[:, 3]).copy()
        data[:, 0] = data[:, 0] - (w / 2.)
        data[:, 1] = data[:, 1] - (data[:, 3] / 2.)
        data[:, 2] = w
    if is_list: return data.tolist()
    return data

def xyxy_to_xyah(data: Union[np.ndarray, List[float], List[List[float]]]):
    return xywh_to_xyah(xyxy_to_xywh(data))

def xyah_to_xyxy(data: Union[np.ndarray, List[float], List[List[float]]]):
    return xywh_to_xyxy(xyah_to_xywh(data))