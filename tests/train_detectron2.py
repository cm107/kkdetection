from kkdetection.detectron2.detector import Detector
from kkdetection.util.com import get_args

if __name__ == "__main__":
    args = get_args()
    detector = Detector(**args.autofix())
    if args.get("preview", None, False):
        detector.preview_augmentation([0,1])
    if args.get("train", None, False):
        detector.train()
    if args.get("infer") is not None:
        file = args.get("infer")
        detector.draw_annotation(file, resize=1000, show=True)
        coco = detector.to_coco(file)
        print(coco.df_json)
        print(coco.df_json.iloc[0])