#!/bin/bash
set -eu

# training
python -i train_ppdet.py --config_url_path picodet/picodet_s_320_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams \
    --coco_json_path ./img/coco.json --image_root ./img/ --batch_size 1 --epoch 10
# training + validation
python -i train_ppdet.py --config_url_path picodet/picodet_s_320_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams \
    --coco_json_path ./img/coco.json --image_root ./img/ --batch_size 1 --epoch 100 \
    --coco_json_path_valid ./img/coco.json --image_root_valid ./img/ --batch_size_valid 1
# training GPU + fp16
python -i train_ppdet.py --config_url_path picodet/picodet_s_320_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams \
    --coco_json_path ./img/coco.json --image_root ./img/ --batch_size_train 1 --epoch 10 ---use_gpu ---fp16
# inference
python -i train_ppdet.py --config_url_path picodet/picodet_s_320_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams \
    --img /home/share/10.git/kkdetection/tests/img/img_dog_cat.jpg