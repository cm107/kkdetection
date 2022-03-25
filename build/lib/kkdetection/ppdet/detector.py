import os, datetime
from typing import List, Union
import cv2
from numpy import isin
import paddle
from ppdet.engine import Trainer, set_random_seed
from ppdet.core.workspace import load_config
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.core.workspace import create
from ppdet.data.source.dataset import DetDataset

# local package
from kkdetection.ppdet.config import download_config_files, CreatePPDetYaml
from kkdetection.ppdet.dataset import ImageDataset
from kkannotation.util.image import draw_annotation
from kkdetection.util.com import correct_dirpath, check_type_list
from kkdetection.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "Detector",
]


BASE_CONFIGS_DIR="./configs/"
COCO_CLASSES=[
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


class Detector(Trainer):
    def __init__(
        self, 
        # base config file
        config_url_path: str="ppyolo/ppyolo_r50vd_dcn_1x_coco.yml",
        base_configs_dir: str=BASE_CONFIGS_DIR, num_classes: int=None,
        weights: str="https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams",
        # train coco dataset
        coco_json_path: str=None, image_root: str=None,
        # train params
        batch_size: int=1, epoch: int=1,
        # validation coco dataset
        coco_json_path_valid: str=None, image_root_valid: str=None,
        # validation params
        batch_size_valid: int=1,
        # other parameters
        use_gpu: bool=False, random_seed: int=0, is_override: bool=False, worker_num: int=1,
        outdir: str=f"./output{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        **kwargs
    ):
        assert isinstance(outdir, str)
        assert isinstance(random_seed, int) and random_seed >= 0
        assert isinstance(num_classes, int) and num_classes > 0
        assert isinstance(worker_num, int) and worker_num >= 0
        outdir = correct_dirpath(outdir)
        # download base config files
        basefile = download_config_files(config_url_path, basedir=base_configs_dir, is_override=is_override)[0]
        # add config setting
        yaml_add = CreatePPDetYaml()
        yaml_add.set_base("../" + basefile)
        # set train params
        mode = "test"
        if isinstance(coco_json_path, str):
            assert isinstance(image_root, str)
            mode = "train"
            yaml_add.set_train_dataset(image_root, coco_json_path)
            yaml_add.set_train_batchsize(batch_size)
            yaml_add.set_epoch(epoch)
        # set validation params
        self.is_validate = False
        if isinstance(coco_json_path_valid, str):
            assert isinstance(image_root_valid, str)
            self.is_validate = True
            yaml_add.set_eval_dataset(image_root_valid, coco_json_path_valid)
            yaml_add.set_eval_batchsize(batch_size_valid)
        # other parameters
        yaml_add.set_num_classes(num_classes)
        yaml_add.set_worker_num(worker_num)
        self.worker_num = worker_num
        if isinstance(outdir, str): yaml_add.set_save_dir(outdir)
        yaml_add.set_weights(weights)
        yaml_add.set_pretrain_weights(weights)
        yaml_add.set_use_gpu(use_gpu)
        paddle.distributed.init_parallel_env()
        set_random_seed(random_seed)
        # kwargs set
        for x, y in kwargs.items():
            try: getattr(yaml_add, f"set_{x}")(y)
            except AttributeError: pass
        # create config file
        filename = outdir + "myyaml.yml"
        os.makedirs(outdir, exist_ok=True)
        yaml_add.save(filename)
        cfg = load_config(filename)
        # check
        if cfg.use_gpu: paddle.set_device('gpu')
        else:           paddle.set_device('cpu')
        if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
            cfg['norm_type'] = 'bn'
        check_config(cfg)
        check_gpu(cfg.use_gpu)
        check_version()
        super().__init__(cfg, mode=mode)
        if isinstance(weights, str): self.load_weights(weights)
    
    def predict(self, data):
        outs = self.model(data)
        for key in ['im_shape', 'scale_factor', 'im_id']:
            outs[key] = data[key]
        for key, value in outs.items():
            if hasattr(value, 'numpy'):
                outs[key] = value.numpy()
        return outs

    def predict_dataloader(self, dataloader):
        self.model.eval()
        results = []
        for step_id, data in enumerate(dataloader):
            self.status['step_id'] = step_id
            outs = self.predict(data)
            results.append(outs)
        return results

    def draw_annotation(self, img: str, threshold: float=0.7, classes: List[str]=COCO_CLASSES, is_show: bool=False):
        assert isinstance(img, str)
        output = self.predict(img)[0]
        img    = cv2.imread(img)
        for class_id, score, x1, y1, x2, y2 in output["bbox"].astype(float):
            if score < threshold: continue
            class_id      = int(class_id)
            catecory_name = str(class_id) if classes is None else classes[class_id]
            img = draw_annotation(
                img, [x1, y1, x2-x1, y2-y1], catecory_name=catecory_name, color_id=class_id
            )
        if is_show:
            cv2.imshow(__name__, img)
            cv2.waitKey(0)
        return img

    def train(self):
        super().train(validate=self.is_validate)
