import numpy as np
import cv2
from ppdet.core.workspace import create
from kkdetection.bytetrack import BYTETracker
from kkdetection.ppdet.detector import Detector
from kkdetection.ppdet.dataset import VideoDataset
from kkannotation.streamer import Recorder
from kkannotation.util.image import draw_annotation
from kkdetection.util.image import xyxy_to_xywh
from kkdetection.util.com import get_args


if __name__ == "__main__":
    args     = get_args()
    detector = Detector(**args.autofix())
    dataset = VideoDataset()
    dataset.set_data(args.get("video"), start_frame_id=0, max_frames=10, step=2)
    recorder = Recorder("./output_tracking.mp4", fps=dataset.streamer.get_fps(), width=dataset.streamer.shape()[1], height=dataset.streamer.shape()[0])
    tracker  = BYTETracker(
        dataset.streamer.shape()[0], dataset.streamer.shape()[1], 
        thre_bbox_high=0.8, thre_bbox_low=0.2,
        thre_iou_high=0.8, thre_iou_low=0.8, thre_iou_new=0.8,
        max_time_lost=50
    )
    target_class_id = args.get("target", int, 0)
    dataloader = create("TestReader")(dataset, 0)
    outputs     = detector.predict_dataloader(dataloader)
    outputs     = [x["bbox"] for x in outputs]
    list_bboxes = [x[x[:, 0] == target_class_id][:, 2:].astype(float) for x in outputs]
    list_scores = [x[x[:, 0] == target_class_id][:, 1 ].astype(float) for x in outputs]
    tracks      = tracker.tracking(list_bboxes, list_scores)
    for i_frame, dict_output in enumerate(tracks):
        frame = dataset.streamer[i_frame]
        for track_id, bbox in dict_output.items():
            frame = draw_annotation(frame, xyxy_to_xywh([int(x) for x in bbox]), catecory_name=str(track_id), color_id=track_id)
        if args.get("is_show", None, False):
            cv2.imshow(__name__, frame)
            cv2.waitKey(0)
        if recorder is not None:
            recorder.write(frame)
    recorder.close()
