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

import random
import torch
from torchvision import transforms
import os
import numpy as np
import cv2
import imageio
import copy
from dataio.utils import nyu40_to_scannet20, color_encoding_nyu40, color_encoding_scannet20, get_frames_of_a_scene, get_multi_views_of_a_scene
from dataio.transforms_base import get_transforms
from dataio.transforms_multiview import get_transforms_multiview
from networks.MobileNet import PIXEL_MEAN, PIXEL_STD


# Multi-view dataset used for training
class ScannetMultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, scannet_root, scene_list, scene_views_root="image_pairs/multi_view",
                 window_size=3, step=1, skip=20, data_aug=True, depth_err=0.0,
                 phase="train", load_all=False, load_label=True, clean_data=False,
                 transform=True, H=480, W=640, seg_classes="scannet20"):
        
        self.scannet_root = scannet_root
        self.H = H
        self.W = W
        self.seg_classes = seg_classes
        self.normalize = transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        if self.seg_classes.lower() == "nyu40":
            self.num_classes = 41
        elif self.seg_classes.lower() == "scannet20":
            self.num_classes = 21
        else:
            raise NotImplementedError
        self.color_encoding = self.get_color_encoding()
        self.data_aug = data_aug
        if not data_aug:
            phase = "test"
        assert phase in ["train", "test"], "Invalid phase type!!! Only support train and test"

        if transform:
            self.transforms = get_transforms_multiview(height=self.H, 
                                                       width=self.W,
                                                       depth_err=depth_err,
                                                       phase=phase)
        else:
            self.transforms = None

        # whether load everything or not, such as rgb_raw, K, c2w, etc.
        self.load_all = load_all
        self.load_label = load_label
        self.window_size = window_size
        self.skip = skip
        self.scene_list = scene_list

        if clean_data:
            self.scene_views_dir = os.path.join(scene_views_root, "skip_{}/{}_views_step_{}/filtered".format(self.skip, window_size, step))
        else:
            self.scene_views_dir = os.path.join(scene_views_root, "skip_{}/{}_views_step_{}/all".format(self.skip, window_size, step))
        self.clean_data = clean_data

        # get paths to all the images, depths and labels
        self.rgb_paths = []
        self.depth_paths = []
        self.label_paths = []
        self.frame_ids = []
        self.c2w_list = []
        self.w2c_list = []
        self.K_rgb_list = []  # [3, 3]
        self.K_depth_list = []  # [3, 3]
        for scene in scene_list:
            scene_dir = os.path.join(self.scannet_root, scene)
            rgb_dir = os.path.join(scene_dir, "color")
            depth_dir = os.path.join(scene_dir, "depth")
            label_dir = os.path.join(scene_dir, "label-{}".format(self.H))
            pose_dir = os.path.join(scene_dir, "pose")
            intrinsic_dir = os.path.join(scene_dir, "intrinsic")
            # intrinsics should be the same for all the frames in the window
            K_rgb = np.loadtxt(os.path.join(intrinsic_dir, "intrinsic_color.txt"))[:3, :3].astype(np.float32)
            K_depth = np.loadtxt(os.path.join(intrinsic_dir, "intrinsic_depth.txt"))[:3, :3].astype(np.float32)

            # load all the views in the window [n_frames, n_views]
            multi_view_file = os.path.join(self.scene_views_dir, "{}.txt".format(scene))
            multi_views = get_multi_views_of_a_scene(multi_view_file)
            for vs in multi_views:
                self.frame_ids.append(vs[self.window_size // 2])
                self.K_rgb_list.append(K_rgb)
                self.K_depth_list.append(K_depth)

                if self.data_aug:
                    random.shuffle(vs)

                # Force the middle view be reference view
                # vs.insert(0, vs.pop(self.window_size // 2))

                # n_views
                self.rgb_paths.append([os.path.join(rgb_dir, "{}.jpg".format(v)) for v in vs])
                self.depth_paths.append([os.path.join(depth_dir, "{}.png".format(v)) for v in vs])
                self.label_paths.append([os.path.join(label_dir, "{}.png".format(v)) for v in vs])

                # if not (os.path.exists(rgb_file) and os.path.exists(depth_file) and os.path.exists(label_file)):
                #     continue

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

    def load_data(self, idx):
        # RGB [H, W, 3]
        rgb_paths, depth_paths = self.rgb_paths[idx], self.depth_paths[idx]

        # rescale rgb intrinsic
        rgb_example = np.array(imageio.imread(rgb_paths[0])).astype(np.float32)  # [H, W, 3]
        H_orig, W_orig, _ = rgb_example.shape
        s = float(H_orig) / float(self.H)
        K_rgb = copy.deepcopy(self.K_rgb_list[idx])
        K_rgb[0, :] /= s
        K_rgb[1, :] /= s
        # rescale depth instrinsic
        depth_example = np.array(imageio.imread(depth_paths[0])).astype(np.float32) / 1000.0
        H_orig, W_orig = depth_example.shape
        s = float(H_orig) / float(self.H)
        K_depth = copy.deepcopy(self.K_depth_list[idx])
        K_depth[0, :] /= s
        K_depth[1, :] /= s

        # load rgbs and peths
        rgbs, depths, depths_test = [], [], []
        for n in range(self.window_size):
            rgb = np.array(imageio.imread(rgb_paths[n])).astype(np.float32)  # [H, W, 3]
            rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
            rgbs.append(rgb)
            depth = np.array(imageio.imread(depth_paths[n])).astype(np.float32) / 1000.0
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)  # [H, W]
            depths.append(depth)

        rgbs = np.stack(rgbs, 0)  # [N, H, W, 3]
        depths = np.stack(depths, 0)
        if len(depths_test) > 0:
            depths_test = np.stack(depths_test, 0)

        sample = {
            "rgb": rgbs,  # [N, H, W, 3]
            "depth": depths,  # [N, H, W]
            "c2w": np.stack(self.c2w_list[idx], 0),  # [N, 4, 4]
            "w2c": np.stack(self.w2c_list[idx], 0),  # [N, 4, 4]
            "K_rgb": K_rgb,  # [3, 3]
            "K_depth": K_depth,  # [3, 3]
        }

        # load label
        if self.load_label:
            label_paths = self.label_paths[idx]
            labels = []
            for label_path in label_paths:
                label = np.array(imageio.imread(label_path))
                if self.seg_classes == "scannet20":
                    label = nyu40_to_scannet20(label)
                # in case there are some invalid labels, but this shouldn't happen?
                label[label > 20] = 0
                labels.append(label)
            labels = np.stack(labels)  # [N, H, W]
            sample["label"] = labels

        if self.load_all:
            sample["frame_id"] = self.frame_ids[idx]
            sample["rgb_raw"] = rgbs
        
        return sample

    def __getitem__(self, idx):
        sample = self.load_data(idx)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.rgb_paths)


# 2D Single-view dataset
class ScannetDataset(torch.utils.data.Dataset):
    def __init__(self, scannet_root, scene_file, scene_views_root="image_pairs/single_view",
                 data_aug=True, phase="train", load_all=False, load_label=True,
                 clean_data=False, scene_id=-1, H=240, W=320, H_test=None, W_test=None,
                 seg_classes="scannet20"):
        self.scannet_root = scannet_root
        self.scene_file = scene_file
        self.H = H
        self.W = W
        self.H_test = H_test
        self.W_test = W_test
        self.seg_classes = seg_classes
        self.normalize = transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
        if self.seg_classes.lower() == "nyu40":
            self.num_classes = 41
        elif self.seg_classes.lower() == "scannet20":
            self.num_classes = 21
        else:
            raise NotImplementedError
        self.color_encoding = self.get_color_encoding()
        if not data_aug:
            phase = "test"
        assert phase in ["train", "test"], "Invalid phase type!!! Only support train and test"
        self.transforms = get_transforms(height=self.H, width=self.W,
                                         height_test=self.H_test,
                                         width_test=self.W_test,
                                         phase=phase)
        # whether load everything or not, such as rgb_raw, K, c2w, etc.
        self.load_all = load_all
        self.load_orig_size = (phase == "test") and (H_test is not None) and (W_test is not None)
        self.load_label = load_label

        # get scenes to be loaded
        scene_list = []
        with open(self.scene_file, "r") as f:
            scenes = f.readlines()
            for scene in scenes:
                scene = scene.strip()  # remove \n
                scene_list.append(scene)
        if scene_id != -1:
            scene_list = [scene_list[scene_id]]
        self.scene_list = scene_list

        # load data from scenes
        if clean_data:
            self.scene_views_dir = os.path.join(scene_views_root, "filtered")
        else:
            self.scene_views_dir = os.path.join(scene_views_root, "all")
        self.clean_data = clean_data
        # get paths to all the images, depths and labels
        self.rgb_paths = []
        self.depth_paths = []
        self.label_paths = []
        self.frame_ids = []
        self.c2w_list = []
        self.K_rgb_list = []  # of original image size
        self.K_depth_list = []
        for scene in scene_list:
            scene_dir = os.path.join(self.scannet_root, scene)
            rgb_dir = os.path.join(scene_dir, "color")
            depth_dir = os.path.join(scene_dir, "depth")
            if self.load_orig_size:
                label_dir = os.path.join(scene_dir, "label-orig")
            else:
                label_dir = os.path.join(scene_dir, "label-{}".format(self.H))
            pose_dir = os.path.join(scene_dir, "pose")
            intrinsic_dir = os.path.join(scene_dir, "intrinsic")
            K_rgb = np.loadtxt(os.path.join(intrinsic_dir, "intrinsic_color.txt"))[:3, :3].astype(np.float32)
            K_depth = np.loadtxt(os.path.join(intrinsic_dir, "intrinsic_depth.txt"))[:3, :3].astype(np.float32)
            frames = get_frames_of_a_scene(os.path.join(self.scene_views_dir, "{}.txt".format(scene)))
            frames = list(range(0, frames[-1], 10)) + [frames[-1]]
            for i in frames:
                rgb_file = os.path.join(rgb_dir, "{}.jpg".format(i))
                depth_file = os.path.join(depth_dir, "{}.png".format(i))
                label_file = os.path.join(label_dir, "{}.png".format(i))
                pose_file = os.path.join(pose_dir, "{}.txt".format(i))
                if not (os.path.exists(rgb_file) and os.path.exists(depth_file)):
                    continue

                self.rgb_paths.append(rgb_file)
                self.depth_paths.append(depth_file)
                self.frame_ids.append(i)
                c2w = np.loadtxt(pose_file).astype(np.float32).reshape(4, 4)
                self.c2w_list.append(c2w)
                self.K_rgb_list.append(K_rgb)
                self.K_depth_list.append(K_depth)

                if self.load_label and os.path.exists(label_file):
                    self.label_paths.append(label_file)

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
        # load rgb
        rgb = np.array(imageio.imread(rgb_path)).astype(np.float32)  # [H, W, 3]
        H_orig, W_orig, _ = rgb.shape
        s = float(H_orig) / float(self.H)
        K_rgb = copy.deepcopy(self.K_rgb_list[idx])
        K_rgb[0, :] /= s
        K_rgb[1, :] /= s
        # TODO: should remove this? others are using INTER_LINEAR
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)

        depth = np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0
        H_orig, W_orig = depth.shape
        s = float(H_orig) / float(self.H)
        K_depth = copy.deepcopy(self.K_depth_list[idx])
        K_depth[0, :] /= s
        K_depth[1, :] /= s
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)  # [H, W]

        sample = {
            "rgb": rgb,
            "depth": depth,
        }

        # load label
        if self.load_label:
            label_path = self.label_paths[idx]
            label = np.array(imageio.imread(label_path))
            if self.seg_classes == "scannet20":
                label = nyu40_to_scannet20(label)
            # in case there are some invalid labels, but this shouldn't happen?
            label[label > 20] = 0
            sample["label"] = label

        if self.load_all:
            sample["frame_id"] = self.frame_ids[idx]
            sample["rgb_raw"] = rgb
            sample["K_rgb"] = K_rgb
            sample["K_depth"] = K_depth
            sample["c2w"] = self.c2w_list[idx]
            sample["w2c"] = np.linalg.inv(self.c2w_list[idx])

        sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.rgb_paths)