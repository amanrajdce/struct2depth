ckpt_dir="/viscompfs/users/amanraj/struct2depth/experiment"
data_dir="/viscompfs/users/tedyu/kitti_processed/" # Set for KITTI
seg_data_dir="/viscompfs/users/amanraj/data/kitti_eigen_instance/" # Set for KITTI
imagenet_ckpt="/viscompfs/users/tedyu/struct2depth/resnet_pretrained/model.ckpt"
#pretrained_ckpt="/home/amanraj/codes/models/research/struct2depth/models/model"

CUDA_VISIBLE_DEVICES=0 python train.py \
  --logtostderr \
  --checkpoint_dir $ckpt_dir \
  --data_dir $data_dir \
  --seg_data_dir $seg_data_dir \
  --architecture resnet \
  --imagenet_ckpt $imagenet_ckpt \
  --batch_size 4 \
  --summary_freq 1 \
  #--pretrained_ckpt $pretrained_ckpt
  #--imagenet_norm true \

  #0000004451-fseg.png;
