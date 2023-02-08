# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import os

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from PIL import Image
from classes_mapping import std_labels, stuff2std

# constants
WINDOW_NAME = "mask2former demo"
d3_40_colors_rgb: np.ndarray = np.array(
    [
        [1, 1, 1],
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input_root", type=str
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def map_by_dict(arr, mapping_dict):
    # NOTE: check missing meta
    # missing_key = set(np.unique(arr))-mapping_dict.keys()
    # for k in missing_key:
    #     mapping_dict[k] = -100

    return np.vectorize(mapping_dict.get)(arr)

def save_semantic_observation(pano_seg_raw, total_frames, colors_map, raw_labels, out_dir):
    out_path = out_dir + '/sem_mask2former'
    os.makedirs(out_path, exist_ok=True)

    pano_seg, segments_info = pano_seg_raw
    pano_seg = pano_seg.cpu().numpy()
    msk = pano_seg == 0

    ins_id2ins = {0: -100}
    ins_id2label = {0: -100}

    for info in segments_info:
        ins_id2ins[info['id']] = info['id']
        std_label = stuff2std[raw_labels[info['category_id']]]
        if std_label is None:
            ins_id2label[info['id']] = 0
            ins_id2ins[info['id']] = 0
            print('\n', raw_labels[info['category_id']], '\n')
        else:
            remapped_cat_id = std_labels.index(std_label)
            ins_id2label[info['id']] = remapped_cat_id
        # if info['isthing']:
        #     ins_id2label[info['id']] = info['category_id']
        # else:
        #     ins_id2label[info['id']] = info['category_id'] + 80 # offset 80 for thing class
    
    label_obs = map_by_dict(pano_seg, ins_id2label)
    ins_i_obs = map_by_dict(pano_seg, ins_id2ins) 

    np.save(out_path+'/%d-label.npy'% total_frames, label_obs)
    np.save(out_path+'/%d-instance.npy'% total_frames, ins_i_obs)

    label_obs_rgb = colors_map[label_obs]
    label_obs_rgb[msk] = [0,0,0]
    semantic_img = Image.fromarray(label_obs_rgb.astype(np.uint8))
    semantic_img.save(out_path+"/%d-label.png" % total_frames)
    
    ins_i_obs = ins_i_obs % 40 + 1
    ins_i_obs[msk] = 0
    ins_i_obs = ins_i_obs.flatten() 
    semantic_img_ins = Image.new("P", (pano_seg.shape[1], pano_seg.shape[0]))
    semantic_img_ins.putpalette(d3_40_colors_rgb.flatten())
    semantic_img_ins.putdata((ins_i_obs).astype(np.uint8))
    semantic_img_ins.save(out_path+"/%d-ins.png" % total_frames)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    meta = demo.metadata

    thing_colors = meta.thing_colors
    stuff_colors = meta.stuff_colors

    raw_labels = meta.stuff_classes

    # colors_map = np.array(thing_colors + stuff_colors)
    colors_map = np.array(stuff_colors)

    file_list = [file for file in os.listdir(args.input_root) if file.endswith('.png')]
    
    out_root = args.input_root.rsplit('/', 1)[0]
    for file in tqdm.tqdm(file_list):
        # use PIL, to be consistent with evaluation
        tot_id = int(file.split('.')[0])

        path = os.path.join(args.input_root, file)
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions = demo.run_on_image(img)
        
        sem_seg = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
        
        save_semantic_observation(predictions["panoptic_seg"], tot_id, colors_map, raw_labels, out_dir=out_root)

        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        # if args.output:
        #     if os.path.isdir(args.output):
        #         assert os.path.isdir(args.output), args.output
        #         out_filename = os.path.join(args.output, os.path.basename(path))
        #     else:
        #         assert len(args.input) == 1, "Please specify a directory with args.output"
        #         out_filename = args.output
        #     visualized_output.save(out_filename)
        # else:
        #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        #     if cv2.waitKey(0) == 27:
        #         break  # esc to quit

