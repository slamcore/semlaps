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
import random
import numpy as np
import torch

from typing import Iterator, Sized
from dataio.utils import color_encoding_scannet20, color_encoding_nyu40


class CustomSampler(torch.utils.data.Sampler):
    r"""Samples scenes s.t. the number of segments distributes more uniformly.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized, n_bins=10) -> None:
        self.data_source = data_source
        self.n_bins = n_bins  # maybe simply hardcode it for now

    def __iter__(self) -> Iterator[int]:
        # define 10 bins
        total_num_samples = len(self.data_source)
        avg_nums_per_bin = total_num_samples // self.n_bins
        remainder = total_num_samples - avg_nums_per_bin * self.n_bins
        num_samples_per_bin = [avg_nums_per_bin] * self.n_bins
        all_bin_indices = list(range(self.n_bins))
        random.shuffle(all_bin_indices)
        plus_one_indices = all_bin_indices[:remainder]
        for plus_one_index in plus_one_indices:
            num_samples_per_bin[plus_one_index] += 1

        # create iterable list of indices
        all_indices_list = list(range(len(self.data_source)))
        # list of bins (indices)
        bins = []
        start_index = 0
        for num_samples in num_samples_per_bin:
            bin = all_indices_list[start_index:start_index + num_samples]
            random.shuffle(bin)
            bins.append(bin)
            start_index += num_samples

        # sample from each bin
        samples_indices = []
        for i in range(avg_nums_per_bin):
            samples_indices += [bin[i] for bin in bins]

        # add remaining indices
        for bin in bins:
            if len(bin) > avg_nums_per_bin:
                samples_indices.append(bin[-1])

        return iter(samples_indices)

    def __len__(self) -> int:
        # 1201
        return len(self.data_source)


class SegmentDatasetTrain(torch.utils.data.Dataset):
    def __init__(self,
                 label_fusion_dir,
                 segment_suffix,
                 scene_list,
                 k=32,
                 data_aug=True,
                 trans=3.5,
                 theta=2 * np.pi,
                 feat_type="prob",
                 use_xyz=True,  # use geometric features or not
                 seg_classes="scannet20"):
        """
        :param segments_root_dir: label_fusion/train
        :param scene_file: scene split file
        :param data_aug:
        :param type: train/val
        :param scene_id: in case we only want to do inference for a single scene
        :param seg_classes:
        """
        self.data_dir = label_fusion_dir
        self.scene_list = scene_list
        self.seg_classes = seg_classes
        self.color_encoding = self.get_color_encoding()
        self.segment_suffix = segment_suffix
        self.feat_type = feat_type
        self.use_xyz = use_xyz
        self.k = k

        # parameters for data augmentation
        self.data_aug = data_aug
        self.theta = theta
        self.trans = trans

        # Need to construct K-NN on-the-fly
        self.seg_center_list = []
        self.seg_cov_list = []
        self.seg_label_list = []
        self.seg_feat_list = []
        self.knn_mat_list = []
        num_segs = []
        for scene in scene_list:
            scene_dir = os.path.join(self.data_dir, scene, self.segment_suffix)
            seg_center = torch.load(os.path.join(scene_dir, "seg_center.pth")).float()
            self.seg_center_list.append(seg_center)  # [N_seg, 3]
            num_segs.append(seg_center.shape[0])
            if os.path.exists(os.path.join(scene_dir, "seg_cov.pth")):
                cov_file = os.path.join(scene_dir, "seg_cov.pth")
            elif os.path.exists(os.path.join(scene_dir, "seg_conv.pth")):
                cov_file = os.path.join(scene_dir, "seg_conv.pth")
            else:
                raise NotImplementedError
            self.seg_cov_list.append(torch.load(cov_file).float())  # [N_seg, 3, 3]
            self.seg_label_list.append(torch.load(os.path.join(scene_dir, "seg_label.pth")).long())  # [N_seg,]

            # input feature
            if self.feat_type == "prob":
                feats = torch.load(os.path.join(scene_dir, "seg_feat_prob.pth")).float()  # [N_seg, C]
            elif self.feat_type == "label":
                feats = torch.load(os.path.join(scene_dir, "seg_feat_label.pth")).float()  # [N_seg, C]
            else:
                raise NotImplementedError

            if not self.use_xyz:
                feats = feats[:, 9:]
            self.seg_feat_list.append(feats)
            # knn_mat
            knn_mat = torch.load(os.path.join(scene_dir, "nn_mat.pth")).long()[:, :k]  # [N_seg, K]
            self.knn_mat_list.append(knn_mat)

        # sort the lists based on number of segments
        sort_indices = sorted(range(len(num_segs)), key=num_segs.__getitem__)
        self.seg_center_list = [self.seg_center_list[i] for i in sort_indices]
        self.seg_cov_list = [self.seg_cov_list[i] for i in sort_indices]
        self.seg_label_list = [self.seg_label_list[i] for i in sort_indices]
        self.seg_feat_list = [self.seg_feat_list[i] for i in sort_indices]
        self.knn_mat_list = [self.knn_mat_list[i] for i in sort_indices]

    def __len__(self):
        return len(self.seg_label_list)

    def train_set(self, indices):
        """
        Collate function for dataloader
        :param indices: List of sampled indices
        :return: locs [1, N_seg_batch, 3],
                 covs [1, N_seg_batch, 3, 3]
                 feats [1, N_seg_batch, C],
                 labels [1, N_seg_batch],
                 knn_indices [N_seg_batch, K],
        """
        locs = []
        covs = []
        feats = []
        labels = []
        knn_indices = []
        points_stored = 0
        for i, idx in enumerate(indices):
            loc = self.seg_center_list[idx]  # [N_seg, 3]
            cov = self.seg_cov_list[idx]  # [N_seg, 3, 3]
            feat = self.seg_feat_list[idx]  # [N_seg, C]
            knn_mat = self.knn_mat_list[idx] + points_stored  # [N_seg, K]
            label = self.seg_label_list[idx]  # [N_seg,]
            n_seg = loc.shape[0]

            # data augmentation
            if self.data_aug:
                # rotation
                theta = np.random.uniform(-self.theta, self.theta)
                rot = torch.tensor([[np.cos(theta), np.sin(theta), 0.],
                                    [-np.sin(theta), np.cos(theta), 0.],
                                    [0., 0., 1.]], dtype=torch.float32)
                loc = (rot @ loc.permute(1, 0)).permute(1, 0)  # [N_seg, 3]
                cov = torch.bmm(rot.unsqueeze(0).repeat(n_seg, 1, 1), cov)
                cov = torch.bmm(cov, rot.permute(1, 0).unsqueeze(0).repeat(n_seg, 1, 1))
                feat[:, 3:6] = (rot @ feat[:, 3:6].permute(1, 0)).permute(1, 0)  # [N_seg, 3]
                feat[:, 6:9] = (rot @ feat[:, 6:9].permute(1, 0)).permute(1, 0)  # [N_seg, 3]

                # translation
                dx = np.random.uniform(-self.trans, self.trans)
                dy = np.random.uniform(-self.trans, self.trans)
                dz = np.random.uniform(-0.1, 0.1)
                trans = torch.tensor([[dx, dy, dz]], dtype=torch.float32)  # [1, 3]
                loc = loc + trans
                feat[:, 6:9] = feat[:, 6:9] + trans

            locs.append(loc)
            covs.append(cov)
            feats.append(feat)
            knn_indices.append(knn_mat)
            labels.append(label)
            points_stored += n_seg  # We need to apply offsets to segment indices

        locs = torch.cat(locs, dim=0).unsqueeze(0)  # [1, N_seg_batch, 3]
        covs = torch.cat(covs, dim=0).unsqueeze(0)  # [1, N_seg_batch, 3, 3]
        feats = torch.cat(feats, dim=0).unsqueeze(0)  # [1, N_seg_batch, C]
        knn_indices = torch.cat(knn_indices, dim=0)  # [N_seg_batch, K]
        labels = torch.cat(labels, dim=0).unsqueeze(0)  # [1, N_seg_batch]

        # return data point
        data = {
            "locs": locs,
            "covs": covs,
            "feats": feats,
            "knn_indices": knn_indices,
            "labels": labels,
        }
        return data

    def get_color_encoding(self):
        if self.seg_classes.lower() == 'nyu40':
            return color_encoding_nyu40
        elif self.seg_classes.lower() == 'scannet20':
            return color_encoding_scannet20
        else:
            raise NotImplementedError

    def get_dataloader(self, batch_size=8, num_workers=8, use_custom_sampler=True, drop_last=True):
        psuedo_dataset = list(range(self.__len__()))
        if use_custom_sampler:
            print("Using custom sampler!!!")
            sampler = CustomSampler(psuedo_dataset)
            shuffle = False
        else:
            print("Not using custom sampler!!!")
            sampler = None
            shuffle = True

        dataloader = torch.utils.data.DataLoader(
            psuedo_dataset,
            batch_size=batch_size,
            collate_fn=self.train_set,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle)
        return dataloader


class SegmentDatasetTest(torch.utils.data.Dataset):
    def __init__(self,
                 label_fusion_dir,  # bayesian-fused data
                 segment_suffix,  # segments-data
                 scene_list,  # list of scenes
                 dataset_type="scannet",
                 k=32,
                 feat_type="prob",
                 load_label=True,
                 use_xyz=True,  # use xyz in feature or not.
                 seg_classes="scannet20"):
        """
        :param segments_root_dir: label_fusion/train
        :param scene_file: scene split file
        :param scene_id: in case we only want to do inference for a single scene
        :param seg_classes:
        """

        assert dataset_type in ["scannet", "slamcore"], "Unknown dataset type!!!"
        self.dataset_type = dataset_type
        self.data_dir = label_fusion_dir
        self.seg_classes = seg_classes
        self.color_encoding = self.get_color_encoding()
        self.segment_suffix = segment_suffix
        self.feat_type = feat_type
        self.use_xyz = use_xyz
        self.k = k
        self.load_label = load_label
        self.scene_list = scene_list
        
        # Need to construct K-NN on-the-fly
        self.seg_center_list = []
        self.seg_cov_list = []
        self.seg_label_list = []
        self.seg_feat_list = []
        self.knn_mat_list = []
        for scene in self.scene_list:
            scene_dir = os.path.join(self.data_dir, scene, self.segment_suffix)
            self.seg_center_list.append(torch.load(os.path.join(scene_dir, "seg_center.pth")).float())  # [N_seg, 3]
            
            if os.path.exists(os.path.join(scene_dir, "seg_cov.pth")):
                cov_file = os.path.join(scene_dir, "seg_cov.pth")
            else:
                raise NotImplementedError
            self.seg_cov_list.append(torch.load(cov_file).float())  # [N_seg, 3, 3]

            if self.load_label:
                self.seg_label_list.append(torch.load(os.path.join(scene_dir, "seg_label.pth")).long())  # [N_seg,]

            # input feature
            if self.feat_type == "prob":
                feats = torch.load(os.path.join(scene_dir, "seg_feat_prob.pth")).float()  # [N_seg, C]
            elif self.feat_type == "label":
                feats = torch.load(os.path.join(scene_dir, "seg_feat_label.pth")).float()  # [N_seg, C]
            else:
                raise NotImplementedError

            if not self.use_xyz:
                feats = feats[:, 9:]
            self.seg_feat_list.append(feats)

            # knn_mat
            knn_mat = torch.load(os.path.join(scene_dir, "nn_mat.pth")).long()[:, :k]  # [N_seg, K]
            self.knn_mat_list.append(knn_mat)

    def __len__(self):
        return len(self.seg_center_list)

    def val_set(self, indices):
        """
        Collate function for dataloader
        :param indices: List of sampled indices
        :return: locs [1, N_seg_batch, 3],
                 covs [1, N_seg_batch, 3, 3]
                 feats [1, N_seg_batch, C],
                 labels [1, N_seg_batch],
                 knn_indices [N_seg_batch, K],
        """
        locs = []
        covs = []
        feats = []
        labels = []
        knn_indices = []
        points_stored = 0
        for i, idx in enumerate(indices):
            loc = self.seg_center_list[idx]  # [N_seg, 3]
            cov = self.seg_cov_list[idx]  # [N_seg, 3, 3]
            feat = self.seg_feat_list[idx]  # [N_seg, C]
            knn_mat = self.knn_mat_list[idx] + points_stored  # [N_seg, K]
            n_seg = loc.shape[0]

            locs.append(loc)
            covs.append(cov)
            feats.append(feat)
            knn_indices.append(knn_mat)
            points_stored += n_seg  # We need to apply offsets to segment indices

            if self.load_label:
                label = self.seg_label_list[idx]  # [N_seg,]
                labels.append(label)

        locs = torch.cat(locs, dim=0).unsqueeze(0)  # [1, N_seg_batch, 3]
        covs = torch.cat(covs, dim=0).unsqueeze(0)  # [1, N_seg_batch, 3, 3]
        feats = torch.cat(feats, dim=0).unsqueeze(0)  # [1, N_seg_batch, C]
        knn_indices = torch.cat(knn_indices, dim=0)  # [N_seg_batch, K]

        # return data point
        data = {
            "locs": locs,
            "covs": covs,
            "feats": feats,
            "knn_indices": knn_indices
        }

        if self.load_label:
            labels = torch.cat(labels, dim=0).unsqueeze(0)  # [1, N_seg_batch]
            data["labels"] = labels

        return data

    def get_color_encoding(self):
        if self.seg_classes.lower() == 'nyu40':
            return color_encoding_nyu40
        elif self.seg_classes.lower() == 'scannet20':
            return color_encoding_scannet20
        else:
            raise NotImplementedError

    def get_dataloader(self, num_workers=4, shuffle=False):
        dataloader = torch.utils.data.DataLoader(
            list(range(self.__len__())),
            batch_size=1,
            collate_fn=self.val_set,
            num_workers=num_workers,
            shuffle=shuffle)
        return dataloader
