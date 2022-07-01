import os, datetime
from typing import List, Union
import cv2
import numpy as np
import paddle
from ppdet.engine import Trainer, set_random_seed
from ppdet.core.workspace import load_config
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.core.workspace import create

# local package
from kkdetection.ppdet.config import download_config_files, CreatePPDetYaml
from kkdetection.ppdet.dataset import ImageDataset
from kkannotation.util.image import draw_annotation
from kkdetection.util.com import correct_dirpath
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
        base_configs_dir: str=BASE_CONFIGS_DIR, num_classes: int=None, num_joints: int=None,
        weights: str="https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams",
        # train coco dataset
        coco_json_path: str=None, image_root: str=None, is_kpt: bool=False,
        # train params
        batch_size: int=1, epoch: int=1, autoaug: bool=False, learning_rate: float=None, snapshot_epoch: int=100,
        # validation coco dataset
        coco_json_path_valid: str=None, image_root_valid: str=None,
        # validation params
        batch_size_valid: int=1,
        # detection parameter
        nms_threshold: float=None, score_threshold: float=None, n_bboxes: int=None, class_name: str="PicoHead",
        # other parameters
        use_gpu: bool=False, random_seed: int=0, is_override: bool=False, worker_num: int=1, outdir: str=None,
        **kwargs
    ):
        logger.info("START")
        assert outdir is None or isinstance(outdir, str)
        assert isinstance(random_seed, int) and random_seed >= 0
        assert isinstance(num_classes, int) and num_classes > 0
        assert num_joints is None or (isinstance(num_joints, int) and num_joints > 0)
        assert isinstance(worker_num, int) and worker_num >= 0
        # download base config files
        if os.path.exists(config_url_path):
            basefile = config_url_path
        else:
            basefile = download_config_files(config_url_path, basedir=base_configs_dir, is_override=is_override)[0]
        cfgbase  = load_config(basefile)
        # add config setting
        yaml_add = CreatePPDetYaml()
        yaml_add.set_base("../" + basefile)
        # set train params
        mode = "test"
        yaml_add.set_worker_num(worker_num)
        self.worker_num = worker_num
        if is_kpt:
            assert num_joints is not None
            yaml_add.set_keypoint_paramter(num_joints, cfgbase["pixel_std"], cfgbase["train_height"], cfgbase["train_width"], cfgbase["hmsize"])
        if isinstance(coco_json_path, str):
            assert isinstance(image_root, str)
            assert learning_rate is not None
            mode = "train"
            yaml_add.set_train_dataset(image_root, coco_json_path, is_kpt=is_kpt)
            yaml_add.set_train_reader(batch_size, autoaug=autoaug, is_kpt=is_kpt)
            yaml_add.set_epoch(epoch)
            yaml_add.set_learning_rate(learning_rate)
            yaml_add.set_snapshot_epoch(snapshot_epoch)
        else:
            yaml_add.set_test_reader(batch_size, worker_num)
        # output directory
        if outdir is None:
            if   mode == "train":
                outdir=f"./output{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            elif mode == "test":
                outdir = "./output_test/"
        outdir = correct_dirpath(outdir)
        # set validation params
        self.is_validate = False
        if isinstance(coco_json_path_valid, str):
            assert isinstance(image_root_valid, str)
            self.is_validate = True
            yaml_add.set_eval_dataset(image_root_valid, coco_json_path_valid, is_kpt=is_kpt)
            yaml_add.set_eval_batchsize(batch_size_valid)
        # set head parameter
        if nms_threshold is not None or score_threshold is not None or n_bboxes is not None:
            yaml_add.set_head_parameer(nms_threshold=nms_threshold, score_threshold=score_threshold, n_bboxes=n_bboxes, class_name=class_name)
        # other parameters
        yaml_add.set_num_classes(num_classes)
        yaml_add.set_save_dir(outdir)
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
        os.makedirs(outdir, exist_ok=True)
        filename = outdir + "myyaml.yml"
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
        logger.info("END")
    
    def predict(self, data: Union[dict, str]):
        logger.info("START")
        self.model.eval()
        # self.model.train()
        # self.model.training = False
        if isinstance(data, str):
            dataset = ImageDataset()
            dataset.set_data([data, ])
            dataloader = create("TestReader")(dataset, 0)
            for x in dataloader:
                data = x
                break
        outs = self.model(data)
        for key in ['im_shape', 'scale_factor', 'im_id', 'crop_bbox']:
            if key in data:
                outs[key] = data[key]
        for key, value in outs.items():
            if hasattr(value, 'numpy'):
                outs[key] = value.numpy()
        logger.info("END")
        return outs
    
    def predict_dataset(self, dataset, batch_size: int=1, is_only_bboxes: bool=False, is_only_keypoint: bool=False):
        logger.info("START")
        assert isinstance(batch_size, int) and batch_size >= 1
        self.model.eval()
        inputs, outputs = {"im_id": [], "image": [], "im_shape": [], "scale_factor": [], "crop_bbox": []}, []
        for i_data in range(len(dataset)):
            data = dataset[i_data]
            for x in inputs.keys():
                if x in data:
                    inputs[x].append(data[x])
            if (len(inputs["im_id"]) >= batch_size) or (i_data + 1 == len(dataset)):
                inputs   = {x: np.stack(y) for x, y in inputs.items() if len(y) > 0}
                inputs   = {x: paddle.to_tensor(y) for x, y in inputs.items()}
                output   = self.predict(inputs)
                outputs += [output, ]
                inputs = {"im_id": [], "image": [], "im_shape": [], "scale_factor": [], "crop_bbox": []}
        if is_only_bboxes:
            bboxes = []
            for i in range(len(outputs)):
                bboxes += np.array_split(outputs[i]["bbox"], outputs[i]["bbox_num"].cumsum()[:-1])
            outputs = bboxes
        if is_only_keypoint:
            kpts = []
            for i in range(len(outputs)):
                output = outputs[i]["keypoint"]
                for ndf in output:
                    ndfwk = ndf[0][:, :, :2] + outputs[i]["crop_bbox"][:, :2].reshape(-1, 1, 2)
                    ndfwk = np.concatenate([ndfwk, ndf[0][:, :, 2:]], axis=-1)
                    kpts += [ndfwk.copy(), ]
            outputs = kpts
        logger.info("END")
        return outputs

    def predict_dataloader(self, dataloader, is_only_bboxes: bool=False, is_only_keypoint: bool=False):
        logger.info("START")
        self.model.eval()
        outputs = []
        for step_id, data in enumerate(dataloader):
            logger.info(f"step: {step_id}")
            self.status['step_id'] = step_id
            outs = self.predict(data)
            outputs.append(outs)
        if is_only_bboxes:
            bboxes = []
            for i in range(len(outputs)):
                bboxes += np.array_split(outputs[i]["bbox"], outputs[i]["bbox_num"].cumsum()[:-1])
            outputs = bboxes
        if is_only_keypoint:
            kpts = []
            for i in range(len(outputs)):
                output = outputs[i]["keypoint"]
                for ndf in output:
                    ndfwk = ndf[0][:, :, :2] + outputs[i]["crop_bbox"][:, :2].reshape(-1, 1, 2)
                    ndfwk = np.concatenate([ndfwk, ndf[0][:, :, 2:]], axis=-1)
                    kpts += [ndfwk.copy(), ]
            outputs = kpts
        logger.info("END")
        return outputs

    def draw_annotation(self, img: str, threshold: float=0.7, classes: List[str]=COCO_CLASSES, is_show: bool=False):
        logger.info("START")
        assert isinstance(img, str)
        output = self.predict(img)
        img    = cv2.imread(img)
        if "bbox" in output:
            for class_id, score, x1, y1, x2, y2 in output["bbox"].astype(float):
                if score < threshold: continue
                class_id      = int(class_id)
                catecory_name = str(class_id) if classes is None else classes[class_id]
                img = draw_annotation(
                    img, bbox=[x1, y1, x2-x1, y2-y1], catecory_name=catecory_name, color_id=class_id
                )
        if "keypoint" in output:
            for outkpt, _ in output["keypoint"]:
                for i_outkpt in outkpt:
                    boolwk = (i_outkpt[:, 2] >= threshold)
                    i_outkpt[boolwk, 2] = 2
                    img = draw_annotation(
                        img, keypoints=[float(x) for x in i_outkpt[boolwk].reshape(-1)],
                    )
        if is_show:
            cv2.imshow(__name__, img)
            cv2.waitKey(0)
        logger.info("END")
        return img

    def train(self):
        logger.info("START")
        super().train(validate=self.is_validate)
        logger.info("END")
