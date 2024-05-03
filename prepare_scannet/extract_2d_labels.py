__copyright__ = """
    SLAMcore Limited
    All Rights Reserved.
    (C) Copyright 2024

    NOTICE:

    All information contained herein is, and remains the property of SLAMcore
    Limited and its suppliers, if any. The intellectual and technical concepts
    contained herein are proprietary to SLAMcore Limited and its suppliers and
    may be covered by patents in process, and are protected by trade secret or
    copyright law.
"""

__license__ = "CC BY-NC-SA 3.0"

import os.path

from _download_scannet_scenes import extract_semantic_label_images
from dataio.utils import get_scene_list
from config import get_scannet_root
from prepare_scannet import util
from tqdm import tqdm
import shutil


# label-processing is based on： https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts
if __name__ == "__main__":
    scannet_root = get_scannet_root()
    scene_list = get_scene_list("configs/scannetv2_trainval.txt")
    label_map_file = os.path.join(os.path.dirname(scannet_root), "scannetv2-labels.combined.tsv")
    label_map = util.read_label_mapping(label_map_file, label_from='id', label_to='nyu40id')
    for scene in tqdm(scene_list):
        extract_semantic_label_images(os.path.join(scannet_root, scene), label_map, skip=20, h=480, w=640)
