#!/bin/bash
set -eu

python tracking_detectron2.py --MODELZOO COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml ---is_bbox_only --video palace.h264.mp4 ---is_show
# ffmpeg -i ./output.mp4 -vcodec libx264 -acodec libmp3lame output_video.mp4