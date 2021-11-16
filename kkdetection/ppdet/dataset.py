import base64
from functools import partial
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
        self.roidbs   = None
    def set_data(
        self, video_file_path: str=None, reverse: bool=False, start_frame_id: int=0, 
        max_frames: int=None, step: int=1
    ):
        self.streamer = Streamer(
            video_file_path, 
            reverse=reverse, start_frame_id=start_frame_id, 
            max_frames=max_frames, step=step
        )
        self.parse_dataset()
    def check_or_download_dataset(self): pass
    def parse_dataset(self):
        if not self.roidbs:
            self.roidbs = [
                {
                    "im_id": np.array([idx]), 
                    "image": cv2.imencode('.png', self.streamer[idx])[1].tobytes(),
                } for idx in range(len(self.streamer))
            ]


class ImageDataset(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def set_data(self, images):
        self.set_images(images)
