import numpy as np
import cv2
from kkdetectron2.bytetrack import BYTETracker
from kkdetectron2.detector import Detector
from kkannotation.streamer import Streamer, Recorder
from kkannotation.util.image import draw_annotation
from kkdetectron2.util.image import xyxy_to_xywh
from kkdetectron2.util.com import get_args


if __name__ == "__main__":
    args     = get_args()
    detector = Detector(**args.autofix())
    streamer = Streamer(args.get("video"), start_frame_id=0, max_frames=None)
    recorder = Recorder("./output_tracking.mp4", fps=streamer.get_fps(), width=streamer.shape()[1], height=streamer.shape()[0])
    tracker  = BYTETracker(
        streamer.shape()[0], streamer.shape()[1], 
        thre_bbox_high=0.8, thre_bbox_low=0.2,
        thre_iou_high=0.8, thre_iou_low=0.8, thre_iou_new=0.8,
        max_time_lost=50
    )
    output, classes = detector.predict(streamer.to_list(), batch_size=15)
    is_target   = [np.isin(x, ["person"]) for x in classes]
    list_bboxes = [x.get("pred_boxes").tensor.numpy()[is_target[i]].astype(float) for i, x in enumerate(output)]
    list_scores = [x.get("scores").           numpy()[is_target[i]].astype(float) for i, x in enumerate(output)]
    tracks      = tracker.tracking(list_bboxes, list_scores)
    for i_frame, dict_output in enumerate(tracks):
        frame = streamer[i_frame]
        for track_id, bbox in dict_output.items():
            frame = draw_annotation(frame, xyxy_to_xywh([int(x) for x in bbox]), catecory_name=str(track_id), color_id=track_id)
        if args.get("is_show", None, False):
            cv2.imshow(__name__, frame)
            cv2.waitKey(0)
        if recorder is not None:
            recorder.write(frame)
    recorder.close()
