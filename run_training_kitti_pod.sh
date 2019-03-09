ckpt_dir="/viscompfs/users/amanraj/data/experiment/training/testing"
data_dir="/viscompfs/users/amanraj/data/kitti_processed/" # Set for KITTI
ins_data_dir="/viscompfs/users/amanraj/data/kitti_eigen_instance/" # Set for KITTI
imagenet_ckpt="/viscompfs/users/amanraj/data/experiment/resnet_pretrained/model.ckpt"
pretrained_ckpt="/viscompfs/users/amanraj/data/experiment/checkpoint/model"

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
