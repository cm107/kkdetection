import os, datetime
from typing import List, Union
import paddle
from ppdet.engine import Trainer, set_random_seed
from ppdet.core.workspace import load_config
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.core.workspace import create

# local package
from kkdetection.ppdet.config import download_config_files, CreatePPDetYaml
from kkdetection.util.com import correct_dirpath, check_type_list
from kkdetection.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "Detector",
]


BASE_CONFIGS_DIR="./configs/"


class Detector(Trainer):
    def __init__(
        self, 
        # base config file
        config_url_path: str="ppyolo/ppyolo_r50vd_dcn_1x_coco.yml",
        base_configs_dir: str=BASE_CONFIGS_DIR, 
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
        use_gpu: bool=False, random_seed: int=0,
        outdir: str=f"./output{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        **kwargs
    ):
        """
        """
        assert isinstance(outdir, str)
        assert isinstance(random_seed, int) and random_seed >= 0
        outdir = correct_dirpath(outdir)
        # download base config files
        basefile = download_config_files(config_url_path, basedir=base_configs_dir)[0]
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
        if isinstance(outdir, str):
            yaml_add.set_save_dir(outdir)
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

    def create_dataloader(self, images: List[str], name_reader: str="TestReader"):
        assert isinstance(images, list) and check_type_list(images, str)
        self.dataset.set_images(images)
        return create(name_reader)(self.dataset, 0)
    
    def predict(self, images: Union[str, List[str]]):
        if not isinstance(images, list): images = [images]
        loader = self.create_dataloader(images)
        self.model.eval()
        results = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            outs = self.model(data)
            for key in ['im_shape', 'scale_factor', 'im_id']:
                outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)
        return results

    def train(self):
        super().train(validate=self.is_validate)
