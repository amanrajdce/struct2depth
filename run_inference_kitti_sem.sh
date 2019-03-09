output_dir="/mnt/1.9TB/struct2depth/experiment/training/semantic_from_scratch/prediction/"
model_checkpoint="/mnt/1.9TB/struct2depth/experiment/training/semantic_from_scratch/model-191121"
input_file="/mnt/1.9TB/kitti_raw/test_files_eigen.txt"
sem_data_dir="/mnt/1.9TB/struct2depth/kitti_test_files_eigen_semantic/"

python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --input_list_file $input_file \
    --sem_data_dir $sem_data_dir \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint \
    --is_semantic true
    #--input_dir $input_dir \
