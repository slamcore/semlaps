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
import matplotlib
import matplotlib.colors
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from networks.MobileNet import PIXEL_MEAN, PIXEL_STD

# Data augmentation operations for single-view datasets


def get_transforms(depth_mean=0.,
                   depth_std=1.,
                   height=None,
                   width=None,
                   height_test=None,
                   width_test=None,
                   phase='train',
                   train_random_rescale=(1.0, 1.4)):
    assert phase in ['train', 'test']

    if phase == 'train':
        transform_list = [
            # This step is actually redundant
            Rescale(height=height, width=width),
            RandomGaussianBlur(),
            RandomRescale(train_random_rescale),
            RandomCrop(crop_height=height, crop_width=width),
            RandomHSV((0.9, 1.1),
                      (0.9, 1.1),
                      (25, 25)),
            RandomFlip(),
            ToTensor(),
            Normalize(depth_mean=depth_mean,
                      depth_std=depth_std),
            # MultiScaleLabel(downsampling_rates=[8, 16, 32])
        ]

    else:
        if height is None and width is None:
            transform_list = []
        else:
            transform_list = [
                Rescale(height=height, width=width, h_test=height_test, w_test=width_test)]
        transform_list.extend([
            ToTensor(),
            Normalize(depth_mean=depth_mean,
                      depth_std=depth_std)
        ])
    transform = transforms.Compose(transform_list)
    return transform


class Rescale:
    def __init__(self, height, width, h_test=None, w_test=None):
        self.height = height
        self.width = width
        self.height_test = h_test
        self.width_test = w_test

    def __call__(self, sample):
        rgb_, depth_ = sample['rgb'], sample['depth']
        rgb = cv2.resize(rgb_, (self.width, self.height),
                         interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth_, (self.width, self.height),
                           interpolation=cv2.INTER_NEAREST)
        sample['rgb'] = rgb
        sample['depth'] = depth

        # in case load different resolution at inference, assert load_all == True
        # only affect rgb_raw, depth, label, inst_label and inst_semantic
        if self.height_test and self.width_test:
            K_depth_test = copy.deepcopy(sample["K_depth"])
            rgb_raw_ = sample["rgb_raw"]
            rgb_raw = cv2.resize(rgb_raw_, (self.width_test, self.height_test),
                                 interpolation=cv2.INTER_LINEAR)
            depth_test = cv2.resize(depth_, (self.width_test, self.height_test),
                                    interpolation=cv2.INTER_NEAREST)
            s = float(self.height_test) / float(self.height)
            K_depth_test[0, :] *= s
            K_depth_test[1, :] *= s
            sample["rgb_raw"] = rgb_raw
            sample['depth_test'] = depth_test
            sample['K_depth_test'] = K_depth_test

            if "label" in sample:
                label_ = sample['label']
                label = cv2.resize(label_, (self.width_test, self.height_test),
                                   interpolation=cv2.INTER_NEAREST)

                sample['label'] = label

            if "inst_label" in sample:
                inst_label_, inst_semantic_ = sample['inst_label'], sample['inst_semantic']
                inst_label = cv2.resize(inst_label_, (self.width_test, self.height_test),
                                        interpolation=cv2.INTER_NEAREST)
                inst_semantic = cv2.resize(inst_semantic_, (self.width_test, self.height_test),
                                           interpolation=cv2.INTER_NEAREST)
                sample['inst_label'] = inst_label
                sample['inst_semantic'] = inst_semantic
        else:
            if "label" in sample:
                label_ = sample['label']
                label = cv2.resize(label_, (self.width, self.height),
                                   interpolation=cv2.INTER_NEAREST)

                sample['label'] = label

            if "inst_label" in sample:
                inst_label_, inst_semantic_ = sample['inst_label'], sample['inst_semantic']
                inst_label = cv2.resize(inst_label_, (self.width, self.height),
                                        interpolation=cv2.INTER_NEAREST)
                inst_semantic = cv2.resize(inst_semantic_, (self.width, self.height),
                                           interpolation=cv2.INTER_NEAREST)

                sample['inst_label'] = inst_label
                sample['inst_semantic'] = inst_semantic

        return sample


class RandomRescale:
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        # TODO: add probability
        rgb, depth, label = sample['rgb'], sample['depth'], sample['label']
        target_scale = np.random.uniform(self.scale_low, self.scale_high)
        # (H, W, C) so all aligned with RGB image size
        target_height = int(round(target_scale * rgb.shape[0]))
        target_width = int(round(target_scale * rgb.shape[1]))
        rgb = cv2.resize(rgb, (target_width, target_height),
                         interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (target_width, target_height),
                           interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (target_width, target_height),
                           interpolation=cv2.INTER_NEAREST)

        sample['rgb'] = rgb
        sample['depth'] = depth
        sample['label'] = label

        if 'inst_label' in sample:
            inst_label, inst_semantic = sample['inst_label'], sample['inst_semantic']
            inst_label = cv2.resize(inst_label, (target_width, target_height),
                                    interpolation=cv2.INTER_NEAREST)
            inst_semantic = cv2.resize(inst_semantic, (target_width, target_height),
                                       interpolation=cv2.INTER_NEAREST)

            sample['inst_label'] = inst_label
            sample['inst_semantic'] = inst_semantic

        return sample


class RandomCrop:
    def __init__(self, crop_height, crop_width, crop_type="rand"):
        self.crop_height = crop_height  # 480
        self.crop_width = crop_width  # 640
        self.rescale = Rescale(self.crop_height, self.crop_width)
        assert crop_type in ["center", "rand"]
        self.crop_type = crop_type

    def __call__(self, sample):
        rgb, depth, label = sample['rgb'], sample['depth'], sample['label']

        h = rgb.shape[0]
        w = rgb.shape[1]
        if h <= self.crop_height or w <= self.crop_width:
            # simply rescale instead of random crop as image is not large enough
            sample = self.rescale(sample)
        else:
            if self.crop_type == "center":
                i, j = (h - self.crop_height) // 2, (w - self.crop_width) // 2
            else:
                i = np.random.randint(0, h - self.crop_height)
                j = np.random.randint(0, w - self.crop_width)
            rgb = rgb[i:i + self.crop_height, j:j + self.crop_width, :]
            depth = depth[i:i + self.crop_height, j:j + self.crop_width]
            label = label[i:i + self.crop_height, j:j + self.crop_width]
            sample['rgb'] = rgb
            sample['depth'] = depth
            sample['label'] = label

            if "inst_label" in sample:
                inst_label, inst_semantic = sample['inst_label'], sample['inst_semantic']
                inst_label = inst_label[i:i +
                                        self.crop_height, j:j+self.crop_width]
                inst_semantic = inst_semantic[i:i +
                                              self.crop_height, j:j+self.crop_width]
                sample['inst_label'] = inst_label
                sample['inst_semantic'] = inst_semantic

        # assert sample["depth"].shape[0] == 240 and sample["depth"].shape[1] == 320, "size incorrect after crop!!!"

        return sample


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, sample):
        if random.random() < 0.5:
            rgb = sample["rgb"]
            rgb = cv2.GaussianBlur(rgb, (self.radius, self.radius), 0)
            sample["rgb"] = rgb
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
        img = sample['rgb']
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
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        sample['rgb'] = img_new

        return sample


class RandomFlip:
    def __call__(self, sample):
        if np.random.rand() > 0.5:
            rgb, depth, label = sample['rgb'], sample['depth'], sample['label']
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()
            # So before this should be a serious bug, depth and label flipped but rgb not
            sample['rgb'] = rgb
            sample['depth'] = depth
            sample['label'] = label

            if "inst_label" in sample:
                inst_label, inst_semantic = sample['inst_label'], sample['inst_semantic']
                inst_label = np.fliplr(inst_label).copy()
                inst_semantic = np.fliplr(inst_semantic).copy()
                sample['inst_label'] = inst_label
                sample['inst_semantic'] = inst_semantic

        return sample


class Normalize:
    def __init__(self, depth_mean, depth_std):
        self._depth_mean = [depth_mean]
        self._depth_std = [depth_std]

    def __call__(self, sample):
        rgb, depth = sample['rgb'], sample['depth']
        rgb = rgb / 255.
        rgb = torchvision.transforms.Normalize(
            mean=PIXEL_MEAN, std=PIXEL_STD)(rgb)

        depth = torchvision.transforms.Normalize(
            mean=self._depth_mean, std=self._depth_std)(depth)

        sample['rgb'] = rgb
        sample['depth'] = depth

        return sample


class ToTensor:
    def __call__(self, sample):
        for key in sample:
            item = sample[key]
            if key == "rgb":
                sample[key] = torch.from_numpy(
                    item.transpose((2, 0, 1))).float()
            elif key == "depth":
                sample[key] = torch.from_numpy(np.expand_dims(item, 0)).float()
            elif key == "label":
                sample[key] = torch.from_numpy(item).long()
            elif key == "frame_id" or key == "scene_id":
                continue
            else:
                if item is not None:
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
