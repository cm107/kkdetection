import os, time, datetime
import numpy as np
import cv2
from typing import List, Tuple, Union
import torch


# detectron2 packages
from detectron2.engine import DefaultTrainer, DefaultPredictor, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog

from fvcore.common.config import CfgNode
import detectron2.utils.comm as comm

# local package
from kkdetection.util.com import MyDict, setattr_deep, check_type, check_type_list, makedirs, correct_dirpath
from kkdetection.util.image import transform_img_from_dataloader
from kkdetection.detectron2.mapper import Mapper
from kkimgaug.util.functions import convert_polygon_to_bool, fit_resize
from kkannotation.coco import CocoManager
from kkdetection.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "Detector",
    "Predictor",
]


DEFAULT_DATASET_NAME = "mydataset"


def set_config(
    args: MyDict=MyDict(), model_zoo_path: str="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml", 
    is_custom: bool=True, cfg: CfgNode=None) -> CfgNode:
    """
    see https://detectron2.readthedocs.io/modules/config.html#detectron2.config.CfgNode
    """
    is_cfg = False
    if cfg is None:
        is_cfg = True
        cfg    = get_cfg()
    for x in args.keys():
        try: setattr_deep(cfg, x, args.get(x, autofix=True))
        except AttributeError: pass
    if is_cfg:
        cfg.merge_from_file(model_zoo.get_config_file(model_zoo_path))
        setattr_deep(cfg, "OUTPUT_DIR",               args.get("OUTPUT_DIR",               str, f"./output{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"))
        if is_custom:
            setattr_deep(cfg, "DATASETS.TRAIN",      (args.get("DATASETS.TRAIN",           str, DEFAULT_DATASET_NAME), ))
            setattr_deep(cfg, "DATASETS.TEST",       (args.get("DATASETS.TEST",            str, DEFAULT_DATASET_NAME), ))
        setattr_deep(cfg, "DATALOADER.NUM_WORKERS",   args.get("DATALOADER.NUM_WORKERS",   int, 4))
        setattr_deep(cfg, "MODEL.WEIGHTS",            args.get("MODEL.WEIGHTS",            str, model_zoo.get_checkpoint_url(model_zoo_path)))
        setattr_deep(cfg, "SOLVER.IMS_PER_BATCH",     args.get("SOLVER.IMS_PER_BATCH",     int, 1))
        setattr_deep(cfg, "INPUT.MIN_SIZE_TRAIN",     args.get("INPUT.MIN_SIZE_TRAIN",     int, 800))
        setattr_deep(cfg, "INPUT.MAX_SIZE_TRAIN",     args.get("INPUT.MAX_SIZE_TRAIN",     int, 1333))
        setattr_deep(cfg, "INPUT.MIN_SIZE_TEST",      args.get("INPUT.MIN_SIZE_TEST",      int, 800))
        setattr_deep(cfg, "INPUT.MAX_SIZE_TEST",      args.get("INPUT.MAX_SIZE_TEST",      int, 1333))
        setattr_deep(cfg, "SOLVER.BASE_LR",           args.get("SOLVER.BASE_LR",           float, 0.001))
        setattr_deep(cfg, "SOLVER.STEPS",             args.get("SOLVER.STEPS",             int, (30000,)))
        setattr_deep(cfg, "SOLVER.MAX_ITER",          args.get("SOLVER.MAX_ITER",          int, 100))
        setattr_deep(cfg, "SOLVER.CHECKPOINT_PERIOD", args.get("SOLVER.CHECKPOINT_PERIOD", int, 5000))
        setattr_deep(cfg, "SOLVER.WARMUP_ITERS",      args.get("SOLVER.WARMUP_ITERS",      int, 1000))
        setattr_deep(cfg, "VIS_PERIOD",               args.get("VIS_PERIOD",               int, 100))
        setattr_deep(cfg, "TEST.DETECTIONS_PER_IMAGE",args.get("TEST.DETECTIONS_PER_IMAGE",int, 200))
    return cfg


def register_catalogs(
    dataset_name: str, coco_json_path: str=None, image_path: str=None, 
    classes: List[str]=None, keypoint_names: List[str]=None, keypoint_flip_map: List[Tuple[str]]=None,
    remove_dataset: bool=False
):
    assert isinstance(dataset_name, str)
    if remove_dataset:
        try: DatasetCatalog.remove(dataset_name) # Cannot re-register without deleting the key
        except KeyError: pass
        try: MetadataCatalog.remove(dataset_name) # Cannot re-register without deleting the key
        except KeyError: pass
    if coco_json_path is not None and image_path is not None:
        assert isinstance(coco_json_path, str)
        assert isinstance(image_path, str)
        coco = CocoManager(coco_json_path)
        if classes is None and isinstance(coco_json_path, str):
            classes = np.sort(coco.df_json["categories_name"].unique()).tolist()
        if keypoint_names is None and isinstance(coco_json_path, str):
            keypoint_names = coco.df_json["categories_keypoints"].iloc[0]
            if np.isnan(keypoint_names) or len(keypoint_names) == 0: keypoint_names = None
        register_coco_instances(dataset_name, {}, coco_json_path, image_path)
    if dataset_name == DEFAULT_DATASET_NAME: assert classes is not None
    if classes is not None:
        assert isinstance(classes, list) and check_type_list(classes, str)
        MetadataCatalog.get(dataset_name).thing_classes = classes
        if keypoint_names is not None:
            assert isinstance(keypoint_names, list) and check_type_list(keypoint_names, str)
            if keypoint_flip_map is not None:
                assert isinstance(keypoint_flip_map, list) and check_type_list(keypoint_flip_map, [list, tuple], str)
                assert len(keypoint_names) == len(keypoint_flip_map)
                assert sum([len(x) == 2 for x in keypoint_flip_map]) == len(keypoint_names)
            else:
                keypoint_flip_map = []
            MetadataCatalog.get(dataset_name).keypoint_names            = keypoint_names
            MetadataCatalog.get(dataset_name).keypoint_flip_map         = keypoint_flip_map
            MetadataCatalog.get(dataset_name).keypoint_connection_rules = [(x[0], x[1], (255,0,0)) for x in keypoint_flip_map] # Use inside Visualizer.
    return classes, keypoint_names


class Detector(DefaultTrainer):
    def __init__(
            self,
            # coco dataset
            coco_json_path: str=None, image_root: str=None,
            # train params
            aug_json_file_path: str=None, resume: bool=False, 
            # validation param
            valid_coco_json_path: List[str]=None, valid_image_root: List[str]=None,
            valid_steps: int=100, valid_ndata: int=1,
            # meta param
            classes: List[str]=None, keypoint_names: List[str]=None, keypoint_flip_map: List[Tuple[str]]=None,
            # others
            is_keyseg: bool=False, is_bbox_only: bool=False,
            # kwargs
            **kwargs
        ):
        """
        Params::
            coco_json_path:
                coco file path of training data.
                If it is not set, it is assumed to be inference only.
            image_root:
                Training data image directory path.
            aug_json_file_path:
                config file path for augmentation
                see: https://github.com/kazukingh01/kkimgaug
            resume:
                If True, We'll resume training from where we left off.
                and Set OUTPUT_DIR of the model you want to restart.
            valid_coco_json_path:
                List of string. Define the validation coco json path you want to add
            valid_image_root:
                validation data image directory paths.
            valid_steps:
                validate steps
            valid_ndata:
                Number of validations to calculate at one time.
                ex) batch 4, valid_ndata 2: 4 * 2 = 8
            classes:
                when training::  If None, determine from coco_json_path.
                when inference:: If None, maybe "coco train 2017"'s classes are set.
                ex) classes=["cat", "dog"]
            keypoint_names:
                when training::  If None, determine from coco_json_path.
                when inference:: If None, maybe "coco train 2017"'s classes are set.
                ex) keypoint_names=["eye", "nose", "mouth"]
            keypoint_flip_map:
                ex) keypoint_flip_map=[["eye", "nose"], ["nose", "mouth"]]
            is_keyseg:
                If True, cfg.MODEL.KEYPOINT_ON = True
            is_bbox_only:
                cfg.MODEL.MASK_ON     = False
                cfg.MODEL.KEYPOINT_ON = False
        Usage::
            >>> from kkdetectron2.detector import Detector
            >>> detector = Detector(MODELZOO="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
            >>> detector.predict("./img/img_dog_cat.jpg")
            {'instances': Instances(num_instances=4, image_height=398, image_width=710, fields=[pred_boxes: Boxes(tensor([[355.5290, 119.3289, 509.9267, 353.4832],
                [110.1354,  87.3155, 348.4151, 343.4511],
                [110.9043,  94.9804, 355.7867, 346.5162],
                [356.3384, 131.8676, 512.8574, 351.3849]], device='cuda:0')), scores: tensor([0.9842, 0.9730, 0.3970, 0.0884], device='cuda:0'), pred_classes: tensor([16, 16, 21, 21], device='cuda:0')])}
            >>> detector.predict_to_df("./img/img_dog_cat.jpg")
                images_id        images_coco_url images_date_captured images_file_name  ... categories_name  categories_skeleton  categories_supercategory  annotations_score
            0          0  ./img/img_dog_cat.jpg  2021-11-01 14:09:52  img_dog_cat.jpg  ...             dog                   []                       dog              0.984
            1          0  ./img/img_dog_cat.jpg  2021-11-01 14:09:52  img_dog_cat.jpg  ...             dog                   []                       dog              0.973
            2          0  ./img/img_dog_cat.jpg  2021-11-01 14:09:52  img_dog_cat.jpg  ...            bear                   []                      bear              0.397
            3          0  ./img/img_dog_cat.jpg  2021-11-01 14:09:52  img_dog_cat.jpg  ...            bear                   []                      bear              0.088
            [4 rows x 26 columns]
        """
        self.predictor      = None
        # coco dataset
        self.args           = MyDict(kwargs)
        self.coco_json_path = coco_json_path
        self.image_root     = image_root
        self.resume         = resume
        # set configuration
        self.cfg            = set_config(
            args=self.args, model_zoo_path=self.args.get("MODELZOO", str, "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"), 
            is_custom=(True if isinstance(self.coco_json_path, str) or isinstance(self.args.get("MODEL.WEIGHTS"), str) else False)
        )
        self.dataset_name   = self.cfg.DATASETS.TRAIN[0]
        classes, keypoint_names = register_catalogs(
            self.dataset_name, self.coco_json_path, self.image_root, 
            classes=classes, keypoint_names=keypoint_names, keypoint_flip_map=keypoint_flip_map,
            remove_dataset = False
        )
        # set FIX configuration
        self.cfg.INPUT.RANDOM_FLIP    = "none"
        self.cfg.SOLVER.WARMUP_METHOD = "linear"
        self.cfg.SOLVER.WARMUP_FACTOR = 1.0 / self.cfg.SOLVER.WARMUP_ITERS
        if is_keyseg:
            self.cfg.MODEL.MASK_ON     = True
            self.cfg.MODEL.KEYPOINT_ON = True
        if is_bbox_only:
            self.cfg.MODEL.MASK_ON     = False
            self.cfg.MODEL.KEYPOINT_ON = False
        self.mapper = None if aug_json_file_path is None else Mapper(self.cfg, config=aug_json_file_path)
        if classes is not None: self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(keypoint_names) if keypoint_names is not None else 0
        self.classes = np.array(MetadataCatalog.get(self.dataset_name).thing_classes)
        if self.coco_json_path is not None:
            # create trainer instance
            super().__init__(self.cfg)
            # validation setting
            if valid_coco_json_path is not None:
                assert check_type_list(valid_coco_json_path, str)
                if valid_image_root is not None:
                    assert check_type_list(valid_image_root, str)
                    assert len(valid_coco_json_path) == len(valid_image_root)
                list_validator: list = []
                for i_valid, json_path in enumerate(valid_coco_json_path): # valid: (dataset_name, json_path, image_path)
                    dataset_name = f"validation{i_valid}"
                    _, _ = register_catalogs(
                        dataset_name, json_path, None if valid_image_root is None else valid_image_root[i_valid], 
                        classes=classes, keypoint_names=keypoint_names, keypoint_flip_map=keypoint_flip_map,
                        remove_dataset = False
                    )
                    list_validator.append(Validator(self.cfg.clone(), dataset_name, trainer=self, steps=valid_steps, ndata=valid_ndata))
                self.register_hooks(list_validator)
        logger.info(f"CLASSES: {self.classes}", color=["BOLD", "GREEN"])

    def build_train_loader(self, cfg) -> torch.utils.data.DataLoader:
        """
        Override. This function is called in super().__init__
        """
        return build_detection_train_loader(cfg, mapper=self.mapper)

    def train(self):
        makedirs(self.cfg.OUTPUT_DIR, exist_ok=True, remake=False)
        self.resume_or_load(resume=self.resume)
        super().train()
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.set_predictor()

    def set_predictor(self):
        self.predictor = Predictor(self.cfg)

    def predict(self, img: Union[str, np.ndarray, List[str], List[np.ndarray]], batch_size: int=1, proc_aug=lambda x: x):
        logger.info(f"predict batch size={batch_size}", color=["BOLD", "CYAN"])
        assert isinstance(batch_size, int) and batch_size > 0
        if self.predictor is None: self.set_predictor()
        if not isinstance(img, list): img = [img]
        if check_type_list(img, str):
            img = [cv2.imread(x) for x in img]
        outputs = []
        for i in range(len(img) // batch_size):
            logger.info(f"predict i batch: {i}")
            batch   = img[i*batch_size : (i+1)*batch_size]
            output  = self.predictor.multi_predict(batch, proc_aug=proc_aug)
            output  = [x["instances"].to("cpu") for x in output]
            outputs = outputs + output
        return outputs, [self.classes[x.get("pred_classes").numpy()] for x in outputs]

    def draw_annotation(self, img: Union[str, np.ndarray], output: dict=None, only_best: bool=False, resize: int=None, show: bool=False, **kwargs):
        import detectron2.utils.visualizer
        detectron2.utils.visualizer._KEYPOINT_THRESHOLD = 0
        if isinstance(img, str): img = cv2.imread(img)
        if output is None:
            output, _ = self.predict(img, **kwargs)
            output    = output[0]
        if only_best: output = output[0:1]
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get(self.dataset_name), 
            scale=1.0, 
            instance_mode=ColorMode.IMAGE #ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
        )
        v   = v.draw_instance_predictions(output)
        img = v.get_image()[:, :, ::-1]
        if resize is not None:
            img = fit_resize(img, "y", resize)
        if show:
            cv2.imshow(__name__, img)
            cv2.waitKey(0)
        return img
    
    def to_coco(
        self, img: Union[str, np.ndarray, List[str], List[np.ndarray]], batch_size: int=1, 
        supercategory: dict=None, save_filepath: str=None, **kwargs
    ) -> CocoManager:
        logger.info(f"to coco format.", color=["BOLD", "CYAN"])
        assert check_type(img, [str, np.ndarray]) or check_type_list(img, [str, np.ndarray])
        img       = img if isinstance(img, list) else [img]
        bool_path = check_type_list(img, str)
        coco      = CocoManager()
        metadata  = MetadataCatalog.get(self.dataset_name)
        outputs   = []
        for img_batch in np.array_split(img, (len(img)//batch_size)+1):
            outputs_batch, _ = self.predict(img_batch.tolist(), batch_size=batch_size, **kwargs)
            for i_img, output in enumerate(outputs_batch):
                imgpath = img_batch[i_img] if bool_path else ""
                height, width = output.image_size
                segmentations, keypoints = None, None
                try: segmentations = output.get("pred_masks").numpy()
                except KeyError: pass
                try: keypoints     = output.get("pred_keypoints").numpy()
                except KeyError: pass
                for i_anno, bbox in enumerate(output.get("pred_boxes").tensor.numpy()):
                    x1, y1, x2, y2 = [float(x) for x in bbox]
                    class_id = output.get("pred_classes")[i_anno]
                    segmentation, keypoint = None, None
                    if segmentations is not None:
                        segmentation = segmentations[i_anno].copy()
                        contours     = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
                        segmentation = [contour.reshape(-1).tolist() for contour in contours]
                        segmentation = [x for x in segmentation if len(x) > 10]
                        segmentation = [[int(x) for x in y] for y in segmentation]
                    if keypoints is not None:
                        keypoint = keypoints[i_anno].copy()
                        keypoint[:, 2] = 1
                        keypoint = keypoint.reshape(-1).tolist()
                    coco.add(
                        imgpath, height, width, (x1, y1, x2 - x1, y2 - y1),
                        metadata.thing_classes[class_id], segmentations=segmentation,
                        keypoints=keypoint, category_name_kpts=metadata.get("keypoint_names")
                    )
            outputs += outputs_batch
        coco.concat_added()
        coco.df_json["annotations_score"] = np.concatenate([x.get("scores").numpy() for x in outputs])
        if keypoints is not None:
            coco.df_json["annotations_score_keypoints"] = [[round(x, 3) for x in y] for y in keypoints[:, :, 2].tolist()]
        if supercategory is not None:
            coco.df_json["categories_supercategory"] = coco.df_json["categories_supercategory"].map(supercategory)
            assert coco.df_json["categories_supercategory"].isna().sum() == 0
        if isinstance(save_filepath, str):
            coco.save(save_filepath)
        return coco

    def preview_augmentation(self, src: Union[int, str, List[int], List[str]], outdir: str="./preview_augmentation", max_images: int=100):
        if isinstance(src, list): assert check_type_list(src, [int, str])
        else: assert check_type(src, [int, str])
        coco   = CocoManager(self.coco_json_path)
        outdir = correct_dirpath(outdir)
        makedirs(outdir, exist_ok=True, remake=True)
        # Select the target coco data.
        if isinstance(src, str):
            coco.df_json = coco.df_json.loc[coco.df_json["images_file_name"] == src]
        elif isinstance(src, int):
            coco.df_json = coco.df_json.iloc[src:src+1]
        else:
            if check_type_list(src, int):
                coco.df_json = coco.df_json.iloc[src, :]
            else:
                coco.df_json = coco.df_json.loc[coco.df_json["images_file_name"].isin(src)]
        org_coco_filepath   = self.coco_json_path
        self.coco_json_path = outdir + "coco.augmentation.json"
        coco.save(self.coco_json_path)
        _, _ = register_catalogs(self.dataset_name, self.coco_json_path, self.image_root, remove_dataset=True)
        if self.mapper is not None: self.mapper.is_preview = True
        super().__init__(self.cfg) # Change only the contents of the coco and create the instance again.
        count = 0
        for i, batch in enumerate(self.data_loader):
            data = batch[0]
            img  = transform_img_from_dataloader(data["image"].detach())
            ## Copy key name from gt_*** to pred_***
            ins = data["instances"].to("cpu")
            if ins.has("gt_boxes"):     ins.set("pred_boxes",     ins.gt_boxes)
            if ins.has("gt_classes"):   ins.set("pred_classes",   ins.gt_classes)
            if ins.has("gt_keypoints"): ins.set("pred_keypoints", ins.gt_keypoints)
            if ins.has("gt_masks"):
                segs = ins.get("gt_masks").polygons
                list_ndf = []
                for seg_a_instance in segs:
                    ndf = convert_polygon_to_bool(img.shape[0], img.shape[1], seg_a_instance)
                    list_ndf.append(ndf)
                ndf = np.concatenate([[ndfwk] for ndfwk in list_ndf], axis=0)
                ins.set("pred_masks", torch.from_numpy(ndf))
            data["instances"] = ins
            img = self.draw_annotation(img, output=data["instances"])
            cv2.imwrite(outdir + "preview_augmentation." + str(i) + ".png", img)
            count += 1
            if count > max_images: break
        self.coco_json_path = org_coco_filepath
        _, _ = register_catalogs(self.dataset_name, self.coco_json_path, self.image_root, remove_dataset=True)
        if self.mapper is not None: self.mapper.is_preview = False
        super().__init__(self.cfg) # Restore the contents of coco and create the instance again.


class Validator(HookBase):
    def __init__(self, cfg: CfgNode, dataset_name: str, trainer: DefaultTrainer, steps: int=10, ndata: int=5):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = (dataset_name, )
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.trainer = trainer
        self.steps = steps
        self.ndata = ndata
        self.loss_dict = {}
        self.data_time = 0
        
    def before_step(self):
        # If you don't put it in "before_step", "storage._latest~~" will be initialized in storage.step after "after_step"
        if self.loss_dict:
            self.trainer._trainer._write_metrics(self.loss_dict, self.data_time)

    def after_step(self):
        if self.trainer.iter > 0 and self.trainer.iter % self.steps == 0:
            list_dict = []
            # self.trainer.model.eval() # This will change the behavior of model(data), so don't do it.
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(self.ndata):
                    data = next(self._loader)
                    loss_dict = self.trainer.model(data)
                    list_dict.append({
                        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                        for k, v in loss_dict.items()
                    })
            loss_dict = {}
            for key in list_dict[0].keys():
                loss_dict[key] = np.mean([dictwk[key] for dictwk in list_dict])
            loss_dict = {
                self.cfg.DATASETS.TRAIN[0] + "_" + k: torch.tensor(v.item()) for k, v in comm.reduce_dict(loss_dict).items()
            }
            self.loss_dict = loss_dict
            self.data_time = time.perf_counter() - start


class Predictor(DefaultPredictor):
    def multi_predict(self, list_images: List[np.ndarray], proc_aug=lambda x: x):
        inputs = []
        with torch.no_grad():
            for original_image in list_images:
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = proc_aug(image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append({"image": image, "height": height, "width": width})
            predictions = self.model(inputs)
            return predictions
