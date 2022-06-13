from ppdet.core.workspace import create
from kkdetection.ppdet.detector import Detector
from kkdetection.ppdet.dataset import VideoDataset
from kkdetection.util.com import get_args


if __name__ == '__main__':
    args     = get_args()
    detector = Detector(**args.autofix())
    # train
    if args.get("coco_json_path") is not None:
        detector.train()
    # inference
    if args.get("img") is not None:
        outputs = detector.predict(args.get("img"))
        detector.draw_annotation(args.get("img"), threshold=0.4, is_show=True)
    # inference video
    if args.get("video") is not None:
        dataset = VideoDataset()
        dataset.set_data(args.get("video"), start_frame_id=0, max_frames=100, step=3)
        dataloader = create("TestReader")(dataset, 0)
        outputs = detector.predict_dataloader(dataloader)
