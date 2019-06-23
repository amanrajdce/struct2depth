# Datasets path
sem_data_dir="/mnt/1.9TB/struct2depth/kitti_test_files_eigen_semantic/"
checkpoint_dir="/mnt/1.9TB/struct2depth/experiment/vanilla"

CUDA_VISIBLE_DEVICES=2 python depth_eval_multiple.py \
    --checkpoint_dir $checkpoint_dir
    #--start_idx 211239
    #--is_semantic \
    #--sem_data_dir $sem_data_dir \
    #--start_idx 211239
