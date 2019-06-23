ckpt_dir="/mnt/1.9TB/struct2depth/experiment/sfmnet/"
data_dir="/mnt/1.9TB/struct2depth/kitti_processed/" # Set for KITTI
ins_data_dir="/mnt/1.9TB/struct2depth/kitti_eigen_instance/" # Set for KITTI
imagenet_ckpt="/mnt/1.9TB/struct2depth/experiment/resnet_pretrained/model.ckpt"
#pretrained_ckpt="/mnt/1.9TB/struct2depth/experiment/checkpoint/model"

CUDA_VISIBLE_DEVICES=2 python train.py \
  --logtostderr \
  --checkpoint_dir $ckpt_dir \
  --data_dir $data_dir \
  --ins_data_dir $ins_data_dir \
  --architecture resnet \
  --batch_size 8 \
  --summary_freq 100 \
  --imagenet_norm true \
  --handle_motion=False \
  --size_constraint_weight=0 \
  --motion_mask=True \
  #--pretrained_ckpt $pretrained_ckpt
  #--learning_rate 0.0002

  #0000004451-fseg.png;
