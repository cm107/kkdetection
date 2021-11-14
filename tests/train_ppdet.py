from kkdetection.ppdet.detector import Detector
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
