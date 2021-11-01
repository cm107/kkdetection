import copy
import torch
import numpy as np
from functools import partial
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from kkimgaug.lib.aug_base import BaseCompose
import kkimgaug.util.procs as P


__all__ = [
    "Mapper"
]


class Mapper(DatasetMapper):
    def __init__(self, *args, config: str=None, is_preview: bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_preview = is_preview
        self.composer   = BaseCompose(
            config, 
            preproc=[
                P.bgr2rgb, 
                P.check_coco_annotations,
                P.bbox_label_auto,
                P.mask_from_polygon_to_bool,
                P.kpt_from_coco_to_xy
            ],
            aftproc=[
                P.rgb2bgr,
                P.mask_inside_bbox,
                P.bbox_compute_from_mask,
                partial(P.mask_from_bool_to_polygon, ignore_n_point=6),
                P.restore_kpt_coco_format,
                self.switch_process,
                P.to_uint8,
            ],
        )
    def switch_process(self, transformed: dict):
        if self.is_preview:
            return P.get_applied_augmentations(transformed, draw_on_image=True)
        else:
            return transformed
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        ##### My Code #####
        transformed = self.composer(
            image=image,
            bboxes=[x["bbox"] if x.get("bbox") else [] for x in dataset_dict["annotations"]],
            mask=[x["segmentation"] if x.get("segmentation") else [] for x in dataset_dict["annotations"]],
            keypoints=[x["keypoints"] if x.get("keypoints") else [] for x in dataset_dict["annotations"]],
        )
        image = transformed["image"]
        if "annotations" in dataset_dict:
            list_annotations = []
            for i, i_instance in enumerate(transformed["label_bbox"]):
                dictwk = dataset_dict["annotations"][i_instance]
                if "bbox" in dictwk:
                    dictwk["bbox"] = transformed["bboxes"][i]
                if "keypoints" in dictwk and len(dictwk["keypoints"]) > 0:
                    dictwk["keypoints"] = transformed["keypoints"][i_instance]
                if "segmentation" in dictwk and len(dictwk["segmentation"]) > 0:
                    dictwk["segmentation"] = transformed["mask"][i]
                list_annotations.append(dictwk)
            dataset_dict["annotations"] = list_annotations
        ##### My Code #####

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
