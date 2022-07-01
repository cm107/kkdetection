import copy
import enum
import cv2
import numpy as np
from typing import List
from ppdet.data.source.dataset import DetDataset, ImageFolder

# local
from kkannotation.streamer import Streamer
from kkdetection.util.com import check_type_list

__all__ = [
    "VideoDataset",
    "KptDataset",
    "ImageDataset",
]


class VideoDataset(DetDataset):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.streamer = None
        self.roidbs   = []
        self._len     = None
        self.mixup_epoch  = -1
        self.cutmix_epoch = -1
        self.mosaic_epoch = -1
    def __len__(self):
        return self._len
    def set_data(
        self, video_file_path: str=None, reverse: bool=False, start_frame_id: int=0, 
        max_frames: int=None, step: int=1
    ):
        if isinstance(self.streamer, Streamer): self.streamer.__del__()
        self.streamer = Streamer(
            video_file_path, 
            reverse=reverse, start_frame_id=start_frame_id, 
            max_frames=max_frames, step=step
        )
        self._len = len(self.streamer)
        self._height, self._width = self.streamer.shape()
        self.parse_dataset(is_force_load=True)
    def check_or_download_dataset(self): pass
    def parse_dataset(self, is_force_load: bool=False):
        if is_force_load or self.roidbs is None:
            self.roidbs = [
                {
                    "im_id": np.array([idx]), 
                    "image": cv2.imencode('.png', copy.deepcopy(self.streamer[idx]) )[1].tobytes(),
                } for idx in range(len(self))
            ]


class KptDataset(DetDataset):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.roidbs   = []
        self.mixup_epoch  = -1
        self.cutmix_epoch = -1
        self.mosaic_epoch = -1
    def __len__(self):
        return len(self.roidbs)
    def set_data(self, frames: List[np.ndarray], bboxes: List[List[int]], scale: float=1.0):
        if isinstance(frames, np.ndarray): frames = [frames, ]
        assert check_type_list(frames, np.ndarray)
        if check_type_list(bboxes, [int, float]): bboxes = [bboxes, ]
        assert check_type_list(bboxes, list, [int, float])
        assert len(frames) == len(bboxes)
        for idx, (frame, bbox) in enumerate(zip(frames, bboxes)):
            frame, bbox = self.crop_image(frame, bbox)
            self.roidbs.append({
                "im_id": np.array([idx]),
                "crop_bbox": np.array(bbox),
                "image": cv2.imencode('.png', copy.deepcopy(frame))[1].tobytes(),
            })
    def check_or_download_dataset(self): pass
    def parse_dataset(self): pass
    @classmethod
    def crop_image(cls, frame: np.ndarray, bbox: List[int], scale: float=1.0):
        x1, y1, x2, y2 = bbox
        cy, cx = (y1 + y2) / 2., (x1 + x2) / 2.
        h , w  = (y2 - y1), (x2 - x1)
        h , w  = h * scale, w * scale
        x1, x2 = cx - (w/2.), cx + (w/2.)
        y1, y2 = cy - (h/2.), cy + (h/2.)
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > frame.shape[1]: x2 = frame.shape[1]
        if y2 > frame.shape[0]: y2 = frame.shape[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return frame[y1:y2, x1:x2, :], [x1, y1, x2, y2, ]


class ImageDataset(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def set_data(self, images):
        self.set_images(images)
