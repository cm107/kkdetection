from kkdetectron2.bytetrack import Tracker
from kkdetectron2.util.com import get_args


if __name__ == "__main__":
    args = get_args()
    tracker = Tracker(**args.autofix())
    tracker.tracking(
        target_classes=args.get("target_classes"), max_count=100, outfilepath="output.mp4", is_draw_prediction=True
    )
