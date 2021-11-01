#!/bin/bash
set -eu
echo "You can run the following sample code !!"
exit 64

# Inference
python train.py ---infer --MODELZOO COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
# Preview augmentation
python train.py --coco_json_path ./img/coco.json --image_root ./img/ --aug_json_file_path ./config.json ---preview --MODELZOO COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml 
# Training Bbox
python train.py --coco_json_path ./img/coco.json --image_root ./img/ --aug_json_file_path ./config.json ---train ---infer --MODELZOO COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
# Inference with trained weight
python train.py ---infer --MODELZOO COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml --OUTPUT_DIR ./outputYYYYMMDDHHMMSS/ --MODEL.WEIGHTS ./outputYYYYMMDDHHMMSS/model_final.pth ----classes cat dog
# Training Bbox with Validation
python train.py --coco_json_path ./img/coco.json --image_root ./img/ --aug_json_file_path ./config.json ---train ---infer --MODELZOO COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml ----valid_coco_json_path ./img/coco.json ----valid_image_root ./img/ --valid_steps 10 --valid_ndata 1
# Training Bbox, learning rate 0.01, batch size 4, max iteration 50
python train.py --coco_json_path ./img/coco.json --image_root ./img/ --aug_json_file_path ./config.json ---train ---infer --MODELZOO COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml --SOLVER.BASE_LR 0.01 --SOLVER.IMS_PER_BATCH 4 --SOLVER.MAX_ITER 50
# Training Bbox + Segmentation
python train.py --coco_json_path ./img/coco.json --image_root ./img/ --aug_json_file_path ./config.json ---train ---infer --MODELZOO COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
# Training Bbox + Segmentation + keypoint
python train.py --coco_json_path ./img/coco.json --image_root ./img/ --aug_json_file_path ./config.json ---train ---infer --MODELZOO COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml ---is_keyseg
