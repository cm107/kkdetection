from kkdetectron2.detector import Detector
from kkdetectron2.util.com import get_args

if __name__ == "__main__":
    args = get_args()
    detector = Detector(**args.autofix())
    if args.get("preview", None, False):
        detector.preview_augmentation([0,1])
    if args.get("train", None, False):
        detector.train()
    if args.get("infer", None, False):
        detector.draw_annoetation("./img/img_dog_cat.jpg", resize=1000, show=True)
        coco = detector.to_coco(["./img/img_dog_cat.jpg"])
