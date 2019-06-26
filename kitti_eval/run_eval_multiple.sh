# Datasets path
sem_data_dir="/mnt/1.9TB/struct2depth/kitti_test_files_eigen_semantic/"
checkpoint_dir="/media/ehdd_2t/amanraj/data/struct2depth/experiment/sfmnet"

CUDA_VISIBLE_DEVICES=2 python depth_eval_multiple.py \
    --checkpoint_dir $checkpoint_dir \
    --start_idx 30174
    #--is_semantic \
    #--sem_data_dir $sem_data_dir \
