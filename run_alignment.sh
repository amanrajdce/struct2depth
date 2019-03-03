data_dir="/mnt/1.9TB/struct2depth/kitti_eigen_instance/" # Set for KITTI
filename="val"

python alignment.py \
  --data_dir $data_dir \
  --file $filename

  #0000004451-fseg.png;
