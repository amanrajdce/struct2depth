ckpt_dir="/mnt/1.9TB/struct2depth/experiment/training/original_from_scratch"
data_dir="/mnt/1.9TB/struct2depth/kitti_processed/" # Set for KITTI
ins_data_dir="/mnt/1.9TB/struct2depth/kitti_eigen_instance/" # Set for KITTI
imagenet_ckpt="/mnt/1.9TB/struct2depth/experiment/resnet_pretrained/model.ckpt"
pretrained_ckpt="/mnt/1.9TB/struct2depth/experiment/checkpoint/model"

python train.py \
  --logtostderr \
  --checkpoint_dir $ckpt_dir \
  --data_dir $data_dir \
  --ins_data_dir $ins_data_dir \
  --architecture resnet \
  --batch_size 4 \
  --summary_freq 100 \
  --imagenet_norm true
  #--pretrained_ckpt $pretrained_ckpt
  #--learning_rate 0.0002
  #--handle_motion=False \
  #--size_constraint_weight=0


  #0000004451-fseg.png;
