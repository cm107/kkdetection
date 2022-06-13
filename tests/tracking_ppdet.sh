#!/bin/bash
set -eu

python -i tracking_ppdet.py --config_url_path ppyolo/ppyolo_r50vd_dcn_1x_coco.yml \
    --weights https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams \
    --video ./palace.h264.mp4 ---is_show --target 0 --num_classes 80
