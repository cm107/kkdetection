#!/bin/bash
set -eu

# training
python train_ppdet.py --config_url_path picodet/picodet_s_320_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams \
    --coco_json_path ./img/coco.json --image_root ./img/ --batch_size 1 --epoch 10 --num_classes 2
# training + validation
python train_ppdet.py --config_url_path picodet/picodet_s_320_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams \
    --coco_json_path ./img/coco.json --image_root ./img/ --batch_size 1 --epoch 10 \
    --coco_json_path_valid ./img/coco.json --image_root_valid ./img/ --batch_size_valid 1 --num_classes 2
# training GPU + fp16
python train_ppdet.py --config_url_path picodet/picodet_s_320_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams \
    --coco_json_path ./img/coco.json --image_root ./img/ --batch_size 1 --epoch 10 ---use_gpu ---fp16
# inference image
python train_ppdet.py --config_url_path picodet/picodet_s_320_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams \
    --img ./img/img_dog_cat.jpg --num_classes 80
# inference video
python train_ppdet.py --config_url_path picodet/picodet_s_320_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams \
    --video ./palace.h264.mp4 --num_classes 80