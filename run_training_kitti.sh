ckpt_dir="/home/amanraj/codes/models/research/struct2depth/experiment"
data_dir="/mnt/1.9TB/struct2depth/kitti_processed/" # Set for KITTI
seg_data_dir="/mnt/1.9TB/struct2depth/kitti_eigen_instance/" # Set for KITTI
imagenet_ckpt="/home/amanraj/codes/models/research/struct2depth/resnet_pretrained/model.ckpt"
pretrained_ckpt="/home/amanraj/codes/models/research/struct2depth/models/model"

python train.py \
  --logtostderr \
  --checkpoint_dir $ckpt_dir \
  --data_dir $data_dir \
  --seg_data_dir $seg_data_dir \
  --architecture resnet \
  --batch_size 4 \
  --summary_freq 10 \
  --imagenet_norm true \
  --pretrained_ckpt $pretrained_ckpt \
  --learning_rate 0.0001
  #--handle_motion=False \
  #--size_constraint_weight=0


  #0000004451-fseg.png;
