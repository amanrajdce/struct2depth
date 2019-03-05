ckpt_dir="/viscompfs/users/amanraj/data/experiment/training/scratch_imgnetckpt"
data_dir="/viscompfs/users/amanraj/data/kitti_processed/" # Set for KITTI
seg_data_dir="/viscompfs/users/amanraj/data/kitti_eigen_instance/" # Set for KITTI
imagenet_ckpt="/viscompfs/users/amanraj/data/experiment/resnet_pretrained/model.ckpt"
pretrained_ckpt="/viscompfs/users/amanraj/data/experiment/checkpoint/model"

CUDA_VISIBLE_DEVICES=0 python train.py \
  --logtostderr \
  --checkpoint_dir $ckpt_dir \
  --data_dir $data_dir \
  --ins_data_dir $ins_data_dir \
  --architecture resnet \
  --imagenet_ckpt $imagenet_ckpt \
  --batch_size 4 \
  --summary_freq 100
  #--pretrained_ckpt $pretrained_ckpt
  #--imagenet_norm true \
  #--learning_rate 0.0002
  #--handle_motion=False \
  #--size_constraint_weight=0
