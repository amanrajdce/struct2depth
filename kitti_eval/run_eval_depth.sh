#!/bin/bash
# Evaluation works on generated .npy files
kitti_dir="/mnt/1.9TB/kitti_raw/" # Raw Kitti directory
pred_dir="/mnt/1.9TB/struct2depth/experiment/training/original_from_scratch/model-191121"
test_file_list="/mnt/1.9TB/kitti_raw/test_files_eigen.txt"

python eval_depth.py \
    --kitti_dir $kitti_dir \
    --pred_dir $pred_dir \
    --test_file_list $test_file_list
