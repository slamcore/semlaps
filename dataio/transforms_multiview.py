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

import copy
import random
import cv2
import imageio
import matplotlib
import matplotlib.colors
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from networks.MobileNet import PIXEL_MEAN, PIXEL_STD

# Data augmentation operations for multi-view dataset


def get_transforms_multiview(depth_mean=0.,
                             depth_std=1.,
                             depth_err=0.0,
                             height=None,
                             width=None,
                             phase='train'):
    assert phase in ['train', 'test']

    if phase == 'train':
        transform_list = [
            RandomRescaleAndCrop(scale=(1.1, 1.4),
                                 crop_height=height,
                                 crop_width=width,
                                 crop_type="rand",
                                 p=0.5),
            RandomFlip(p=0.5),
            RandomGaussianBlur(),
            RandomDepthJittor(err=depth_err, p=0.5),
            RandomHSV((0.9, 1.1),
                      (0.9, 1.1),
                      (25, 25)),
            ToTensor(),
            Normalize(depth_mean=depth_mean,
                      depth_std=depth_std),
            # MultiScaleLabel(downsampling_rates=[8, 16, 32])
        ]

    else:
        transform_list = [
            ToTensor(),
            Normalize(depth_mean=depth_mean,
                      depth_std=depth_std)
        ]
    transform = transforms.Compose(transform_list)
    return transform


class RandomRescaleAndCrop:
    def __init__(self, scale=(1.1, 1.4), crop_height=480, crop_width=640, crop_type="rand", p=0.5):
        self.scale_min, self.scale_max = scale
        self.width, self.height = crop_width, crop_height
        self.crop_type = crop_type
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            rgbs, depths, labels, K_rgb, K_depth = sample['rgb'], sample['depth'], sample['label'], copy.deepcopy(sample["K_rgb"]), copy.deepcopy(sample["K_depth"])
            N, H, W, _ = rgbs.shape
            assert H == self.height and W == self.width, "Dimension doesn't match!!!"
            s = np.random.uniform(self.scale_min, self.scale_max)
            H_ = int(round(s * H))
            W_ = int(round(s * W))

            # scale
            scaled_rgbs, scaled_depths, scaled_labels = [], [], []
            for n in range(N):
                scaled_rgb = cv2.resize(rgbs[n, ...], (W_, H_), interpolation=cv2.INTER_LINEAR)
                scaled_rgbs.append(scaled_rgb)
                scaled_depth = cv2.resize(depths[n, ...], (W_, H_), interpolation=cv2.INTER_NEAREST)
                scaled_depths.append(scaled_depth)
                scaled_label = cv2.resize(labels[n, ...], (W_, H_), interpolation=cv2.INTER_NEAREST)
                scaled_labels.append(scaled_label)
            scaled_rgbs = np.stack(scaled_rgbs, axis=0)  # [N, H', W', 3]
            scaled_depths = np.stack(scaled_depths, axis=0)  # [N, H', W']
            scaled_labels = np.stack(scaled_labels, axis=0)

            # crop
            if self.crop_type == "center":
                i, j = (H_ - H) // 2, (W_ - W) // 2
            else:
                i = np.random.randint(0, H_ - H)
                j = np.random.randint(0, W_ - W)

            for n in range(N):
                rgbs[n, ...] = scaled_rgbs[n, i:i + H, j:j + W, :]
                depths[n, ...] = scaled_depths[n, i:i + H, j:j + W]
                labels[n, ...] = scaled_labels[n, i:i + H, j:j + W]

            # intrinsics
            K_rgb[0, 0] *= s
            K_rgb[1, 1] *= s
            K_rgb[0, 2] = K_rgb[0, 2] * s - j
            K_rgb[1, 2] = K_rgb[1, 2] * s - i
            K_depth[0, 0] *= s
            K_depth[1, 1] *= s
            K_depth[0, 2] = K_depth[0, 2] * s - j
            K_depth[1, 2] = K_depth[1, 2] * s - i

            sample["rgb"] = rgbs
            sample["depth"] = depths
            sample["label"] = labels
            sample["K_rgb"] = K_rgb
            sample["K_depth"] = K_depth

        return sample


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            rgbs, depths, labels, K_rgb, K_depth = sample['rgb'], sample['depth'], sample['label'], copy.deepcopy(sample["K_rgb"]), copy.deepcopy(sample["K_depth"])
            N, H, W, _ = rgbs.shape
            for n in range(N):
                rgbs[n, ...] = np.fliplr(rgbs[n, ...]).copy()
                depths[n, ...] = np.fliplr(depths[n, ...]).copy()
                labels[n, ...] = np.fliplr(labels[n, ...]).copy()

            K_rgb[0, 0] *= -1
            K_rgb[0, 2] = W - 1 - K_rgb[0, 2]
            K_depth[0, 0] *= -1
            K_depth[0, 2] = W - 1 - K_depth[0, 2]
            # So before this should be a serious bug, depth and label flipped but rgb not
            sample['rgb'] = rgbs
            sample['depth'] = depths
            sample['label'] = labels
            sample["K_rgb"] = K_rgb
            sample["K_depth"] = K_depth

            if "inst_label" in sample:
                inst_label, inst_semantic = sample['inst_label'], sample['inst_semantic']
                inst_label = np.fliplr(inst_label).copy()
                inst_semantic = np.fliplr(inst_semantic).copy()
                sample['inst_label'] = inst_label
                sample['inst_semantic'] = inst_semantic

        return sample


class RandomGaussianBlur:
    def __init__(self, radius=5, p=0.5):
        self.radius = radius
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            rgb = sample["rgb"]  # [N, H, W, 3]
            N, H, W, _ = rgb.shape
            new_rgb = np.zeros((N, H, W, 3))
            for i in range(N):
                # TODO: change to in-place operation
                new_rgb[i, ...] = cv2.GaussianBlur(rgb[i, ...], (self.radius, self.radius), 0)
            sample["rgb"] = new_rgb

        return sample

class RandomDepthJittor:
    def __init__(self, err=0.01, p=0.5) -> None:
        self.err = err
        self.p = p    
    def __call__(self, sample):
        if self.err <= 0.0 or self.p <= 0.0:
            return sample
    
        if random.random() < self.p:
            depth = sample["depth"]
            assert depth.ndim == 2 or depth.ndim == 3, "depth dimension must be [H, W] or [N, H, W]!!!"
            if depth.ndim == 2:
                H, W = depth.shape
                depth = depth + self.err * depth * np.random.randn(H, W)
            else:
                N, H, W = depth.shape
                depth = depth + self.err * depth * np.random.randn(N, H, W)
            sample["depth"] = depth
        return sample



class RandomHSV:
    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        if random.random() < 0.5:
            imgs = sample['rgb']  # [N, H, W, 3]
            N, H, W, _ = imgs.shape
            imgs_new = np.zeros((N, H, W, 3))
            for i in range(N):
                img = imgs[i]
                img_hsv = matplotlib.colors.rgb_to_hsv(img)
                img_h = img_hsv[:, :, 0]
                img_s = img_hsv[:, :, 1]
                img_v = img_hsv[:, :, 2]

                h_random = np.random.uniform(min(self.h_range), max(self.h_range))
                s_random = np.random.uniform(min(self.s_range), max(self.s_range))
                v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
                img_h = np.clip(img_h * h_random, 0, 1)
                img_s = np.clip(img_s * s_random, 0, 1)
                img_v = np.clip(img_v + v_random, 0, 255)
                img_hsv = np.stack([img_h, img_s, img_v], axis=2)  # [H, W, 3]
                img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

                imgs_new[i, ...] = img_new

            sample['rgb'] = imgs_new

        return sample


class Normalize:
    def __init__(self, depth_mean, depth_std):
        self._depth_mean = [depth_mean]
        self._depth_std = [depth_std]

    def __call__(self, sample):
        rgb, depth = sample['rgb'], sample['depth']
        rgb = rgb / 255.
        rgb = torchvision.transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)(rgb)

        depth = torchvision.transforms.Normalize(mean=self._depth_mean, std=self._depth_std)(depth)

        sample['rgb'] = rgb
        sample['depth'] = depth

        return sample


class ToTensor:
    def __call__(self, sample):
        for key in sample:
            item = sample[key]
            if key == "rgb":
                sample[key] = torch.from_numpy(item).permute(0, 3, 1, 2).float()  # [N, 3, H, W]
            elif key == "depth":
                sample[key] = torch.from_numpy(np.expand_dims(item, 1)).float()  # [N, 1, H, W]
            elif key == "label":
                sample[key] = torch.from_numpy(item).long()  # [N, H, W]
            elif key == "frame_id" or key == "scene_id":
                continue
            else:
                sample[key] = torch.from_numpy(item).float()

        return sample


class MultiScaleLabel:
    def __init__(self, downsampling_rates=None):
        if downsampling_rates is None:
            self.downsampling_rates = [8, 16, 32]
        else:
            self.downsampling_rates = downsampling_rates

    def __call__(self, sample):
        label = sample['label']

        h, w = label.shape

        sample['label_down'] = dict()

        # Nearest neighbor interpolation
        for rate in self.downsampling_rates:
            label_down = cv2.resize(label.numpy(), (w // rate, h // rate),
                                    interpolation=cv2.INTER_NEAREST)
            sample['label_down'][rate] = torch.from_numpy(label_down)

        return sample


if __name__ == "__main__":
    import os
    from dataio.scannet import ScannetMultiViewDataset
    from dataio.utils import create_label_image, color_encoding_scannet20, get_scene_list
    from config import get_scannet_root
    import open3d as o3d
    H, W = 240, 320
    scannet_root = get_scannet_root()
    scene_split = "../configs/scannetv2_train.txt"
    scene = get_scene_list(scene_split)[0]
    sequence = ScannetMultiViewDataset(scannet_root, [scene], phase="test",
                                       clean_data=False, data_aug=False, transform=False, H=H, W=W)
    save_dir = "vis/data_aug/crop"
    os.makedirs(save_dir, exist_ok=True)
    sample = sequence[20]
    rgbs, depths, labels, K, w2cs = sample["rgb"], sample["depth"], sample["label"], sample["K_depth"], sample["w2c"]

    # TSDF-Fusion before transform
    K_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    voxel_length = 0.02
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=0.04,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    rgbs_vis = []
    labels_vis = []
    for i in range(len(rgbs)):
        rgb, depth, label = rgbs[i], depths[i], labels[i]
        rgbs_vis.append(rgb)
        labels_vis.append(create_label_image(label, color_encoding_scannet20))
        rgb_o3d = o3d.geometry.Image((rgb).astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=1.0,
                                                                  depth_trunc=5.0,
                                                                  convert_rgb_to_intensity=False)
        volume.integrate(rgbd, K_o3d, w2cs[i])

    rgbs_vis = np.concatenate(rgbs_vis, axis=1)
    labels_vis = np.concatenate(labels_vis, axis=1)
    vis_img = np.concatenate([rgbs_vis, labels_vis], axis=0)
    imageio.imwrite(os.path.join(save_dir, "rgb_orig.png"), vis_img)

    print("Extract a triangle mesh from the volume and visualize it.")
    cloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print("Writing file:", "mesh.ply")
    o3d.io.write_triangle_mesh(os.path.join(save_dir, "mesh-fused-orig.ply"), mesh)

    # Test random crop
    Crop = transforms.Compose([RandomRescaleAndCrop(scale=(1.1, 1.4), crop_height=H, crop_width=W, p=1.0),
                               RandomFlip(p=1.0),
                               RandomGaussianBlur(),
                               RandomHSV((0.9, 1.1),
                                         (0.9, 1.1),
                                         (25, 25))])
    sample_t = Crop(sample)
    rgbs, depths, labels, K, w2cs = sample_t["rgb"], sample_t["depth"], sample_t["label"], sample_t["K_depth"], sample_t["w2c"]

    # TSDF-Fusion after transform
    K_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    voxel_length = 0.02
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=0.04,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    rgbs_vis = []
    labels_vis = []
    for i in range(len(rgbs)):
        rgb, depth, label = rgbs[i], depths[i], labels[i]
        rgbs_vis.append(rgb)
        labels_vis.append(create_label_image(label, color_encoding_scannet20))
        rgb_o3d = o3d.geometry.Image((rgb).astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=1.0,
                                                                  depth_trunc=5.0,
                                                                  convert_rgb_to_intensity=False)
        volume.integrate(rgbd, K_o3d, w2cs[i])

    rgbs_vis = np.concatenate(rgbs_vis, axis=1)
    labels_vis = np.concatenate(labels_vis, axis=1)
    vis_img = np.concatenate([rgbs_vis, labels_vis], axis=0)
    imageio.imwrite(os.path.join(save_dir, "rgb_transform.png"), vis_img)

    print("Extract a triangle mesh from the volume and visualize it.")
    cloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print("Writing file:", "mesh.ply")
    o3d.io.write_triangle_mesh(os.path.join(save_dir, "mesh-fused-transform.ply"), mesh)
