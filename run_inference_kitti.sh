#input_dir="/mnt/1.9TB/kitti_raw/2011_09_26/2011_09_26_drive_0084_sync/image_02/data"
output_dir="/mnt/1.9TB/struct2depth/experiment/checkpoint/prediction/"
model_checkpoint="/mnt/1.9TB/struct2depth/experiment/checkpoint/model"
input_file="/mnt/1.9TB/kitti_raw/test_files_eigen.txt"

python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --input_list_file $input_file \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint
    #--input_dir $input_dir \
