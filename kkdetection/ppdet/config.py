import os, yaml, re
from typing import Union, List
import ppdet

# local package
from kkdetection.util.com import download_file, correct_dirpath


__all__ = [
    "download_config_file",
    "CreatePPDetYaml",
]


KEY_BASE = "_BASE_"


def download_config_files(
    config: str, basedir: str="./configs/", 
    baseurl = f"https://raw.githubusercontent.com/PaddlePaddle/PaddleDetection/release/{ppdet.version.full_version[:3]}/configs/",
    is_override: bool=False
):
    """
    config:
        picodet/picodet_s_320_coco.yml
    """
    basedir = correct_dirpath(basedir)
    os.makedirs(basedir, exist_ok=True)
    def work(config, baseurl, basedir, is_override: bool=False):
        url      = baseurl + config
        dirpath  = basedir + os.path.dirname(config)
        filepath = basedir + config
        os.makedirs(dirpath, exist_ok=True)
        if is_override or not os.path.exists(filepath):
            print(f"donwload: {filepath}")
            download_file(url, filepath=filepath)
        else:
            print(f"already exist: {filepath}")
        return filepath
    list_files = [work(config, baseurl, basedir, is_override=is_override)]
    with open(list_files[0]) as file:
        besefile = yaml.load(file, Loader=yaml.Loader)
    if besefile.get(KEY_BASE) is not None:
        addpath = os.path.dirname(config)
        if addpath[-1] != "/": addpath += "/"
        for x in besefile[KEY_BASE]:
            list_files += download_config_files(x, basedir=basedir + addpath, baseurl=baseurl + addpath, is_override=is_override)
    return list_files


class CreatePPDetYaml(object):
    def __init__(self):
        self.yaml  = ""
        self.space = "  "
    def set_base(self, value: str):
        assert isinstance(value, str)
        self.yaml += f"_BASE_: [\n{self.space}'{value}',\n]\n"
    def set_use_gpu(self, value: Union[bool, str]):
        assert isinstance(value, bool) or (isinstance(value, str) and value in ["true", "false", "True", "False"])
        self.yaml += f"use_gpu: {str(value)}\n"
    def set_fp16(self, value: Union[bool, str]):
        assert isinstance(value, bool) or (isinstance(value, str) and value in ["true", "false", "True", "False"])
        self.yaml += f"fp16: {str(value)}\n"
    def set_save_dir(self, value: str):
        assert isinstance(value, str)
        self.yaml += f"save_dir: {value}\n"
    def set_weights(self, value: str):
        assert isinstance(value, str)
        self.yaml += f"weights: {value}\n"
    def set_pretrain_weights(self, value: str):
        assert isinstance(value, str)
        self.yaml += f"pretrain_weights: {value}\n"
    def set_num_classes(self, value: int):
        assert isinstance(value, int) and value > 0
        self.yaml += f"num_classes: {value}\n"
    def set_epoch(self, value: int):
        assert isinstance(value, int) and value > 0
        self.yaml += f"epoch: {value}\n"
    def set_worker_num(self, value: int):
        assert isinstance(value, int) and value >= 0
        self.yaml += f"worker_num: {value}\n"
        self.yaml += f"""
TestReader:
   worker_num: {value}
"""
    @classmethod
    def check_dataset_attr(cls, path_dir: str, path_coco: str):
        assert isinstance(path_dir, str)  and os.path.exists(path_dir)
        assert isinstance(path_coco, str) and os.path.exists(path_coco)
        path_dir = re.sub("/+", "/", path_dir)
        if path_dir[-1] == "/": path_dir = path_dir[:-1]
        image_dir   = path_dir.split("/")[-1]
        dataset_dir = "/".join(path_dir.split("/")[:-1])
        return image_dir, path_coco, dataset_dir
    def set_train_dataset(self, path_dir: str, path_coco: str):
        image_dir, path_coco, dataset_dir = self.check_dataset_attr(path_dir, path_coco)
        self.yaml += f"""
TrainDataset:
  !COCODataSet
    image_dir: {image_dir}
    anno_path: {path_coco}
    dataset_dir: {dataset_dir}
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']
"""
    def set_eval_dataset(self, path_dir: str, path_coco: str):
        image_dir, path_coco, dataset_dir = self.check_dataset_attr(path_dir, path_coco)
        self.yaml += f"""
EvalDataset:
  !COCODataSet
    image_dir: {image_dir}
    anno_path: {path_coco}
    dataset_dir: {dataset_dir}
"""
    def set_test_dataset_image(self, path_dir: str):
        image_dir, _, dataset_dir = self.check_dataset_attr(path_dir, None)
        self.yaml += f"""
TestImageDataset:
  !ImageDataset
    image_dir: {image_dir}
    dataset_dir: {dataset_dir}
"""
    def set_test_dataset_video(self):
        self.yaml += f"""
TestVideoDataset:
  !VideoDataset
"""
    def set_train_batchsize(self, value: int):
        assert isinstance(value, int)
        self.yaml += f"TrainReader:\n{self.space}batch_size: {value}\n"
    def set_eval_batchsize(self, value: int):
        assert isinstance(value, int)
        self.yaml += f"EvalReader:\n{self.space}batch_size: {value}\n"
    def set_test_batchsize(self, value: int):
        assert isinstance(value, int)
        self.yaml += f"TestReader:\n{self.space}batch_size: {value}\n"
    def save(self, filepath: str):
        with open(filepath, mode="w") as f:
            f.write(self.yaml)


class StringYamlController(object):
    def __init__(self):
        self.string = ""
    def __str__(self):
        return self.string
    def __repr__(self):
        return self.__str__()
    def add(self, filepath: str):
        assert isinstance(filepath, str)
        assert os.path.exists(filepath)
        with open(filepath) as f:
            self.string += "".join(f.readlines())
    @classmethod
    def find_segment(cls, string, str_segment: str) -> (str, int, int):
        start_id  = 0
        regex     = f"(\n|) *{str_segment}.*\n"
        search    = re.search(regex, string)
        if search is None:
            raise KeyError(f"'{regex}' is not found.")
        start_id += search.start()
        string    = string[start_id:]
        if string[:1] == "\n":
            string    = string[1:]
            start_id += 1
        nspace_base = re.search("^ *", string).end()
        str_target = ""
        for i, strwk in enumerate(string.split("\n")):
            nspace = re.search("^ *", strwk).end()
            if i > 0 and nspace <= nspace_base and strwk.find(":") >= 0:
                break
            str_target += strwk + "\n"
        return str_target, start_id, start_id + len(str_target)
    def find(self, str_find: str) -> (str, int, int):
        """
        Params::
            str_find: TrainDataset/image_dir
        Return::

        """
        str_find = str_find.split("/")
        string   = self.string
        start_id, end_id = 0, 0
        for x in str_find:
            string, i_st, i_ed = self.find_segment(string, x)
            start_id += i_st
            end_id    = start_id + len(string)
        return string, start_id, end_id
    def set_attr(self, key: str, value: str):
        string, start_id, end_id = self.find(key)
        stid = string.find(":")
        str_before  = self.string[:start_id]
        str_after   = self.string[end_id:]
        str_replace = string[:stid] + ":" + value + "\n"
        self.string = str_before + str_replace + str_after
    def del_attr(self, key: str):
        _, start_id, end_id = self.find(key)
        str_before  = self.string[:start_id]
        str_after   = self.string[end_id:]
        self.string = str_before + str_after
    def save(self, filepath: str):
        assert isinstance(filepath, str)
        with open(filepath, mode="w") as f:
            f.write(self.string)
