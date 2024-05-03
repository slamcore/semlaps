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

import os

import torch
from torchvision import transforms
import numpy as np
import imageio

from dataio.utils import (
    color_encoding_nyu40,
    color_encoding_scannet20
)
from dataio.transforms_base import get_transforms
from dataio.transforms_multiview import get_transforms_multiview
from networks.MobileNet import PIXEL_MEAN, PIXEL_STD


def get_multi_views_of_a_scene_on_the_fly(max_frame_id, start_frame_id=0, window_size=3, skip=20):
    """
    Get a list of multi-view frame_ids on the fly:
    0, 20, 40
    20, 0, 40
    40, 20, 60
    ...
    :return: list of all multi-view frames
    """
    multi_views = []
    frame_ids = np.asarray(list(range(start_frame_id, max_frame_id, skip)))
    for ref_view in frame_ids:
        # let i be ref-view and find n nearest views
        dist = np.abs(frame_ids - ref_view)
        nn_views_id = np.argsort(dist)[1:window_size]
        candidates = [ref_view]
        for nn_id in nn_views_id:
            candidates.append(frame_ids[nn_id])

        assert len(candidates) == window_size, "Length mismatch!!!"
        multi_views.append(candidates)

    return multi_views


# 2D Multi-view dataset
class SLAMcoreMultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, scene, H=480, W=640, window_size=3, center_crop=True, skip=10, load_all=False, seg_classes="scannet20"):
        """
        """
        self.dataset_root = dataset_root
        self.scene = scene
        self.data_dir = os.path.join(self.dataset_root, self.scene)  # dir to a specific sequence
        self.H, self.W = H, W
        self.H_orig, self.W_orig = 480, 848
        self.window_size = window_size
        self.center_crop = center_crop

        self.seg_classes = seg_classes
        self.normalize = transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        if self.seg_classes.lower() == "nyu40":
            self.num_classes = 41
        elif self.seg_classes.lower() == "scannet20":
            self.num_classes = 21
        else:
            raise NotImplementedError
        self.color_encoding = self.get_color_encoding()

        # We do not do any fine-tuning on SLAMcore sequences
        self.phase = "test"
        self.transforms = get_transforms_multiview(phase=self.phase)
        self.load_all = load_all  # whether loading everything or not, such as rgb_raw, K, c2w, etc.

        # get paths to all the images, depths and labels
        self.rgb_paths = []
        self.depth_paths = []
        self.label_paths = []
        self.frame_ids = []
        self.c2w_list = []
        self.w2c_list = []

        rgb_dir = os.path.join(self.data_dir, "color")
        depth_dir = os.path.join(self.data_dir, "depth")
        pose_dir = os.path.join(self.data_dir, "pose")
        self.K = np.loadtxt(os.path.join(self.data_dir, "K.txt"))

        if self.center_crop:  # crop horizontally, such that resolution becomes 4:3
            self.W_crop = int(self.H * 4 / 3)
            self.offset_x = (self.W - self.W_crop) // 2
            self.K[0, 2] -= self.offset_x
        else:
            self.W_crop = self.W
            self.offset_x = 0
        
        # Load multi-view frames
        assert skip >= 10 and skip % 10 == 0, "Only support 10, 20, etc..."
        self.skip = skip
        self.skip_orig = 10  # original skip == 10 for slamcore sequences
        max_frame_id = self.skip_orig * (len(os.listdir(rgb_dir)) - 1)
        multi_views = get_multi_views_of_a_scene_on_the_fly(max_frame_id, window_size=self.window_size, skip=self.skip)

        # get rgb/d filenames for all frames
        for vs in multi_views:
            self.frame_ids.append(vs[0])
            # n_views
            self.rgb_paths.append([os.path.join(rgb_dir, "{}.png".format(v)) for v in vs])
            self.depth_paths.append([os.path.join(depth_dir, "{}.png".format(v)) for v in vs])
            # already filtered out invalid poses
            c2w = [np.loadtxt(os.path.join(pose_dir, "{}.txt".format(v))).astype(np.float32).reshape(4, 4) for v in vs]
            self.c2w_list.append(np.stack(c2w, 0))  # [N, 4, 4]
            w2c = [np.linalg.inv(np.loadtxt(os.path.join(pose_dir, "{}.txt".format(v))).astype(np.float32).reshape(4, 4)) for v in vs]
            self.w2c_list.append(w2c)  # [N, 4, 4]

    def get_color_encoding(self):
        if self.seg_classes.lower() == 'nyu40':
            return color_encoding_nyu40
        elif self.seg_classes.lower() == 'scannet20':
            return color_encoding_scannet20
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        # RGB [H, W, 3]
        rgb_paths, depth_paths = self.rgb_paths[idx], self.depth_paths[idx]
        # load rgbs and depths
        rgbs, depths = [], []
        for n in range(self.window_size):
            # load rgb
            rgb = np.array(imageio.imread(rgb_paths[n])).astype(np.float32)  # [H, W, 3]
            if self.center_crop:
                rgb = rgb[:, self.offset_x:self.offset_x + self.W_crop, :]

            rgbs.append(rgb)

            # load depth
            depth = np.array(imageio.imread(depth_paths[n])).astype(np.float32) / 1000.0
            if self.center_crop:
                depth = depth[:, self.offset_x:self.offset_x + self.W_crop]

            depths.append(depth)

        rgbs = np.stack(rgbs, 0)  # [N, H, W, 3]
        depths = np.stack(depths, 0)  # [N, H, W]

        sample = {
            "rgb": rgbs,  # [N, H, W, 3]
            "depth": depths,  # [N, H, W]
            "c2w": np.stack(self.c2w_list[idx], 0),  # [N, 4, 4]
            "w2c": np.stack(self.w2c_list[idx], 0),  # [N, 4, 4]
            "K": self.K,  # [3, 3]
            "K_depth": self.K
        }

        if self.load_all:
            sample["frame_id"] = self.frame_ids[idx]
            sample["rgb_raw"] = rgbs

        sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.rgb_paths)


# Single-view dataset
class SLAMcoreDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, scene, H=480, W=640, skip=10, load_all=False, seg_classes="scannet20"):
        """
        """
        self.dataset_root = dataset_root
        self.scene = scene
        self.data_dir = os.path.join(self.dataset_root, self.scene)  # dir to a specific sequence
        self.H, self.W = H, W

        assert skip >= 10 and skip % 10 == 0, "Only support 10, 20, etc..."
        self.skip = skip
        self.seg_classes = seg_classes
        self.normalize = transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        if self.seg_classes.lower() == "nyu40":
            self.num_classes = 41
        elif self.seg_classes.lower() == "scannet20":
            self.num_classes = 21
        else:
            raise NotImplementedError
        self.color_encoding = self.get_color_encoding()

        # We do not do any fine-tuning on SLAMcore sequences
        self.phase = "test"
        self.transforms = get_transforms(phase=self.phase)
        self.load_all = load_all  # whether loading everything or not, such as rgb_raw, K, c2w, etc.

        # get paths to all the images, depths and labels
        self.rgb_paths = []
        self.depth_paths = []
        self.label_paths = []
        self.frame_ids = []
        self.c2w_list = []

        rgb_dir = os.path.join(self.data_dir, "color")
        depth_dir = os.path.join(self.data_dir, "depth")
        pose_dir = os.path.join(self.data_dir, "pose")
        frames = sorted([int(fid[:-4]) for fid in os.listdir(rgb_dir)])

        # self.W_crop will always be the REAL width
        self.K = np.loadtxt(os.path.join(self.data_dir, "K"))

        # get rgb/d filenames for all frames
        for i in frames:
            rgb_file = os.path.join(rgb_dir, "{}.png".format(i))
            depth_file = os.path.join(depth_dir, "{}.png".format(i))
            pose_file = os.path.join(pose_dir, "{}.txt".format(i))
            if not (os.path.exists(rgb_file) and os.path.exists(depth_file)):
                continue

            self.rgb_paths.append(rgb_file)
            self.depth_paths.append(depth_file)
            self.frame_ids.append(i)
            c2w = np.loadtxt(pose_file).astype(np.float32).reshape(4, 4)
            self.c2w_list.append(c2w)

    def get_color_encoding(self):
        if self.seg_classes.lower() == 'nyu40':
            return color_encoding_nyu40
        elif self.seg_classes.lower() == 'scannet20':
            return color_encoding_scannet20
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        # RGB [H, W, 3]
        rgb_path, depth_path = self.rgb_paths[idx], self.depth_paths[idx]

        # load rgb [0, 255] float32
        rgb = np.array(imageio.imread(rgb_path)).astype(np.float32)  # [H, W, 3]
        # load depth
        depth = np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0

        sample = {
            "rgb": rgb,
            "depth": depth,
        }

        if self.load_all:
            sample["frame_id"] = self.frame_ids[idx]
            sample["rgb_raw"] = rgb
            sample["K"] = self.K
            sample["K_depth"] = self.K
            sample["c2w"] = self.c2w_list[idx]
            sample["w2c"] = np.linalg.inv(self.c2w_list[idx])

        sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.rgb_paths)
