import copy
import cv2
import numpy as np
from ppdet.data.source.dataset import DetDataset, ImageFolder

# local
from kkannotation.streamer import Streamer


__all__ = [
    "VideoDataset",
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
        self.streamer = Streamer(
            video_file_path, 
            reverse=reverse, start_frame_id=start_frame_id, 
            max_frames=max_frames, step=step
        )
        self._len = len(self.streamer)
        self._height, self._width = self.streamer.shape()
        self.parse_dataset(is_force_load=True)
        self.streamer.__del__()
    def check_or_download_dataset(self): pass
    def parse_dataset(self, is_force_load: bool=False):
        if is_force_load or self.roidbs is None:
            self.roidbs = [
                {
                    "im_id": np.array([idx]), 
                    "image": cv2.imencode('.png', copy.deepcopy(self.streamer[idx]) )[1].tobytes(),
                } for idx in range(len(self))
            ]


class ImageDataset(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def set_data(self, images):
        self.set_images(images)
