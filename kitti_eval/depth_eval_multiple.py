"""
Evaluate depth results on multiple models.

This script is for running evaluation for depth model which is a two step process:
python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --input_list_file $input_file \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint

# Generated .npy files are evaluated
python eval_depth.py \
    --kitti_dir $kitti_dir \
    --pred_dir $pred_dir \
    --test_file_list $test_file_list
"""
import logging
import argparse
import subprocess
import os
import sys
from collections import namedtuple
from eval_depth import main as depth_evaluator

KITTI_DIR="/mnt/1.9TB/kitti_raw/"
INPUT_FILE=KITTI_DIR+"test_files_eigen.txt"
INFERENCE="/home/amanraj/codes/struct2depth/inference.py"

def create_logger(args):
    """
    Create a logging object
    """
    output_dir = os.path.join(args.checkpoint_dir, "prediction")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    hdlr = logging.FileHandler(os.path.join(output_dir, 'evaluation_depth.log'))
    hdlr.setLevel(logging.DEBUG)
    msg_format = '%(asctime)s [%(levelname)s] %(message)s'
    formatter = logging.Formatter(msg_format)
    ch.setFormatter(formatter)
    hdlr.setFormatter(formatter)
    root.addHandler(ch)
    root.addHandler(hdlr)
    logging.info(sys.version_info)
    logging.info(args)
    return logging

def main_call(args, logger):
    # Get all the checkpoint paths
    """
    with open(os.path.join(args.checkpoint_dir, 'checkpoint')) as f:
        files = f.readlines()
    files = [f.split(" ")[-1].strip("\n\"") for f in files if "all_model" in f]
    """
    files = os.listdir(args.checkpoint_dir)
    files = [f.split(".")[0] for f in files if ".meta" in f]
    files = [int(f.split("-")[-1]) for f in files]
    files.sort()
    files = ["model-" + str(f) for f in files]

    pred_dir = os.path.join(args.checkpoint_dir, "prediction")

    for f in files:
        init_chkp = os.path.join(args.checkpoint_dir, f)
        logger.info("Evaluating for: {}".format(init_chkp))
        out_dir = os.path.join(pred_dir, os.path.basename(f))

        global INPUT_FILE
        global INFERENCE
        EVAL_CMD = "python " + INFERENCE + " --file_extension png --depth "
        EVAL_CMD += "--input_list_file " + INPUT_FILE + " --output_dir " + out_dir
        EVAL_CMD += " --model_ckpt " + init_chkp
        if args.is_semantic:
            EVAL_CMD += " --sem_data_dir " + args.sem_data_dir + " --is_semantic true"

        output = subprocess.call(['bash','-c', EVAL_CMD])
        if output != 0:
            logger.error('Evaluation stage-1 failed')
            sys.exit(-1)

        global KITTI_DIR
        arg_list = ['test_file_list', 'kitti_dir', 'pred_dir', 'min_depth', 'max_depth']
        eval_args = namedtuple('eval_args', arg_list)
        pass_eval_args = eval_args(INPUT_FILE, KITTI_DIR, out_dir, 1e-3, 80)
        depth_evaluator(pass_eval_args, logger)
        logger.info("Finished evaluatin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="for parsing arguments for flownet evaluation")
    parser.add_argument(
        '--checkpoint_dir',
        help='directory containing checkpoint models',
        type=str,
        required=True
    )
    parser.add_argument(
        '--is_semantic',
        help='whether to add semantic in input',
        action="store_true"
    )
    parser.add_argument(
        '--sem_data_dir',
        help='directory containing semantic',
        type=str,
        default=None
    )
    args = parser.parse_args()
    logger = create_logger(args)
    main_call(args, logger)
