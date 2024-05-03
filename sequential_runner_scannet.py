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
from collections import deque
import time
import argparse

import numpy as np
import imageio
import cv2
import copy
import trimesh
import open3d as o3d
from tqdm import tqdm
import torch.nn.functional as F
import torch
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from sklearn.neighbors import KDTree
import plyfile
from config import get_scannet_root, get_scannet_test_root, load_yaml
from networks.LatentPriorNetwork import LatentPriorNetwork
from networks.MVCNet import MVCNet
from networks.SegConvNet import SegConvNet
from dataio.scannet import ScannetMultiViewDataset
from dataio.utils import vert_label_to_color, get_scene_list, color_encoding_scannet20, color_encoding_nyu40, nyu40_to_scannet20, create_label_image
from dataio.transforms_base import get_transforms
from metric.iou import IoU
from networks.rend_utils import project_pcd
from qpos.segment_mesh_online_v2 import SegmentationLogger, process_sequence_with_segmenter
from prepare_3d_training_data import compute_normals_o3d


def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()


class ScannetInferenceRunner(object):
    """
    LPN inference
    """
    def __init__(self, logdir, model_name="LPN", skip=20, window_size=3, max_lost=80, H=480, W=640, epoch=19, device=torch.device("cuda")):
        self.scannet_root = get_scannet_root()
        self.device = device
        self.logdir = logdir
        self.PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406])
        self.PIXEL_STD = torch.tensor([0.229, 0.224, 0.225])

        # load pre-trained model
        cfg_file = os.path.join(self.logdir, "config.yaml")
        self.cfg = load_yaml(cfg_file)
        self.cfg.window_size = window_size
        self.model_name = model_name
        self.model = get_model(self.cfg, model_name=model_name, device=self.device)
        ckpt_path = os.path.join(self.logdir, "checkpoints/chkpt-{}.pth".format(epoch))
        pretrained_state_dict = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(pretrained_state_dict["state_dict"])
        self.model.eval()

        # details for the sequence
        self.skip = skip
        self.max_lost = max_lost
        self.window_size = window_size
        self.H = H
        self.W = W
        
        # profiler for timing
        self.profiler = Profiler()
    
    def get_time(self):
        """
        :return: get timing statistics
        """
        torch.cuda.synchronize()
        return time.time()

    @torch.no_grad()
    def run_batched_inference(self, scene, metric=None, save=False, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(self.logdir, "inference_skip{}_window{}/scannet/{}".format(self.skip, self.window_size, scene))

        if save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Offline batched dataset
        scannet_sequence = ScannetMultiViewDataset(self.scannet_root,
                                                            scene,
                                                            phase="test",
                                                            skip=self.skip,
                                                            window_size=self.window_size,
                                                            step=1,
                                                            load_label=True,
                                                            data_aug=False,
                                                            clean_data=False,
                                                            H=self.H, W=self.W,
                                                            H_test=self.H, W_test=self.W,
                                                            load_all=True)
        for data in tqdm(scannet_sequence):
            frame_id = data["frame_id"]
            # label_gt = data["label"]
            rgb, depth, c2w, w2c, K = data["rgb"].to(self.device), \
                data["depth"].to(self.device), \
                data["c2w"].to(self.device), \
                data["w2c"].to(self.device), \
                data["K_depth"].to(self.device)
           
            result_dict = self.model(rgb.unsqueeze(0), depth.unsqueeze(0), K.unsqueeze(0), c2w.unsqueeze(0), w2c.unsqueeze(0))
            out = result_dict["out"]

            if metric is not None:
                label_gt = data["label"][0].unsqueeze(0)
                metric.add(out.detach(), label_gt.detach())

            if save:
                self.save_inference(out, rgb, save_dir, frame_id)
    
    @torch.no_grad()
    def run_causal_inference(self, scene, metric=None, label_only=True, skip_invalid=False, save=False, save_dir=None, load_label=False):
        if save_dir is None:
            save_dir = os.path.join(self.logdir, "causal_inference_skip{}_window{}/scannet/{}".format(self.skip, self.window_size, scene))

        if save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = ScannetScene(scene, skip=self.skip, H=self.H, W=self.W, load_label=load_label)
        # need a queue for storing previous features
        # a frame (frame_id, c2w, w2c, skip1, skip2, bneck_feat)
        mem_q = deque()
        for frame_data in tqdm(data):
            # [1, 3, H, W]
            rgb, depth, c2w, w2c, K, frame_id = frame_data["rgb"].unsqueeze(0).to(self.device), \
                                                frame_data["depth"].unsqueeze(0).to(self.device), \
                                                frame_data["c2w"], \
                                                frame_data["w2c"], \
                                                frame_data["K"].unsqueeze(0).to(self.device), \
                                                frame_data["frame_id"]

            if c2w is not None:
                c2w = c2w.unsqueeze(0).to(self.device)
                w2c = w2c.unsqueeze(0).to(self.device)
                t1 = self.get_time()
                skip1, skip2, feat = self.model.feature_net_forward(rgb, depth_input=depth if self.cfg.use_ssma else None)
                t2 = self.get_time()
                self.profiler.append_encoder(t2 - t1)
                data_tuple = (skip1, skip2, feat, depth, c2w, w2c, frame_id)
                if len(mem_q) > 0 and (frame_id - mem_q[-1][-1]) > self.max_lost:  # lost track, restart
                    mem_q = deque()
                mem_q.append(data_tuple)
                if len(mem_q) > self.window_size:
                    mem_q.popleft()
                out_dict = self.model.causal_forward(mem_q, K, profiler=self.profiler)
                out = out_dict["out"]
            elif not skip_invalid:
                # perform single-view inference
                ret = self.model.single_view_forward(rgb, depth_input=depth if self.cfg.use_ssma else None)
                out = ret["out"]
            else:
                continue
            
            if metric is not None:
                label_gt = frame_data["label"].unsqueeze(0)
                metric.add(out.detach(), label_gt.detach())

            if save:
                self.save_inference(out, rgb, save_dir, frame_id, label_only=label_only)

    @torch.no_grad()
    def run_causal_inference_batch_warp(self, scene, metric=None, skip_invalid=False, save=False, save_dir=None, load_label=False):
        if save_dir is None:
            save_dir = os.path.join(self.logdir, "causal_inference_skip{}_window{}/scannet/{}".format(self.skip, self.window_size, scene))

        if save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = ScannetScene(scene, skip=self.skip, H=self.H, W=self.W, load_label=load_label)
        # need a queue for storing previous features
        # a frame (frame_id, c2w, w2c, skip1, skip2, bneck_feat)
        mem_q = deque()
        for frame_data in tqdm(data):
            # [1, 3, H, W]
            rgb, depth, c2w, w2c, K, frame_id = frame_data["rgb"].unsqueeze(0).to(self.device), \
                                                frame_data["depth"].unsqueeze(0).to(self.device), \
                                                frame_data["c2w"], \
                                                frame_data["w2c"], \
                                                frame_data["K"].unsqueeze(0).to(self.device), \
                                                frame_data["frame_id"]

            if c2w is not None:
                c2w = c2w.unsqueeze(0).to(self.device)
                w2c = w2c.unsqueeze(0).to(self.device)
                t1 = self.get_time()
                skip1, skip2, feat = self.model.feature_net_forward(rgb, depth_input=depth if self.cfg.use_ssma else None)
                t2 = self.get_time()
                self.profiler.append_encoder(t2 - t1)
                data_tuple = (skip1, skip2, feat, depth, c2w, w2c, frame_id)
                if len(mem_q) > 0 and (frame_id - mem_q[-1][-1]) > self.max_lost:  # lost track, restart
                    mem_q = deque()
                mem_q.append(data_tuple)
                if len(mem_q) > self.window_size:
                    mem_q.popleft()
                out_dict = self.model.causal_forward_batch_warp(mem_q, K, profiler=self.profiler)
                out = out_dict["out"]
            elif not skip_invalid:
                # perform single-view inference
                ret = self.model.single_view_forward(rgb, depth_input=depth if self.cfg.use_ssma else None)
                out = ret["out"]
            else:
                continue
            
            if metric is not None:
                label_gt = frame_data["label"].unsqueeze(0)
                metric.add(out.detach(), label_gt.detach())

            if save:
                self.save_inference(out, rgb, save_dir, frame_id)

    @torch.no_grad()
    def run_causal_inference_fused(self, scene, metric=None, skip_invalid=False, load_label=False, save=False, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(self.logdir, "causal_inference_fused_skip{}_window{}/scannet/{}".format(self.skip, self.window_size, scene))

        if save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = ScannetScene(scene, skip=self.skip, H=self.H, W=self.W, load_label=load_label)
        # need a queue for storing previous features
        # a frame (frame_id, c2w, w2c, skip1, skip2, bneck_feat)
        mem_q = deque()
        for frame_data in tqdm(data):
            # [1, C, H, W]
            rgb, depth, c2w, w2c, K, frame_id = frame_data["rgb"].unsqueeze(0).to(self.device), \
                frame_data["depth"].unsqueeze(0).to(self.device), \
                frame_data["c2w"], \
                frame_data["w2c"], \
                frame_data["K"].unsqueeze(0).to(self.device), \
                frame_data["frame_id"]

            if c2w is not None:
                c2w = c2w.unsqueeze(0).to(self.device)
                w2c = w2c.unsqueeze(0).to(self.device)
                t1 = self.get_time()
                skip1, skip2, feat = self.model.feature_net_forward(rgb, depth_input=depth)
                t2 = self.get_time()
                self.profiler.append_encoder(t2 - t1)
                data_tuple = [skip1, skip2, feat, depth, c2w, w2c, frame_id]
                if len(mem_q) > 0 and (frame_id - mem_q[-1][-1]) > self.max_lost:  # lost track, restart
                    mem_q = deque()
                mem_q.append(data_tuple)
                if len(mem_q) > self.window_size:
                    mem_q.popleft()
                out_dict = self.model.causal_forward(mem_q, K, profiler=self.profiler)
                out = out_dict["out"]
                if "skip1" in out_dict:
                    mem_q[-1][0] = out_dict["skip1"]
                    mem_q[-1][1] = out_dict["skip2"]
                    mem_q[-1][2] = out_dict["feat"]
            elif not skip_invalid:
                # perform single-view inference
                ret = self.model.single_view_forward(rgb, depth_input=depth)
                out = ret["out"]
            else:
                continue
            
            if metric is not None:
                label_gt = frame_data["label"].unsqueeze(0)
                metric.add(out.detach(), label_gt.detach())

            if save:
                self.save_inference(out, rgb, save_dir, frame_id)

    @torch.no_grad()
    def run_single_view_inference(self, scene, metric=None, save=False, save_dir=None, skip_invalid=True, load_label=False):
        if save_dir is None:
            save_dir = os.path.join(self.logdir, "singleview_inference/scannet/{}".format(scene))

        if save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = ScannetScene(scene, skip=self.skip, H=self.H, W=self.W, load_all=True, load_label=load_label)
        # need a queue for storing previous features
        # a frame (frame_id, c2w, w2c, skip1, skip2, bneck_feat)
        for frame_data in tqdm(data):
            rgb, depth, c2w, w2c, K, frame_id = frame_data["rgb"].unsqueeze(0).to(self.device), \
                frame_data["depth"].unsqueeze(0).to(self.device), \
                frame_data["c2w"], \
                frame_data["w2c"], \
                frame_data["K"].unsqueeze(0).to(self.device), \
                frame_data["frame_id"]

            if c2w is not None or not skip_invalid:
                # perform single-view inference
                ret = self.model.single_view_forward(rgb, depth_input=depth if self.cfg.use_ssma else None)
                out = ret["out"]
            else:
                continue

            if metric is not None:
                label_gt = frame_data["label"].unsqueeze(0)
                metric.add(out.detach(), label_gt.detach())

            if save:
                self.save_inference(out, rgb, save_dir, frame_id)

    def save_inference(self, out, rgb, save_dir, frame_id, label_only=False):
        output = out[0]
        if label_only:
            image_to_save = np.zeros((self.H, self.W, 3))
            # predicted label
            label_pred = output.argmax(0).cpu().numpy()
            label_pred_image = create_label_image(label_pred, color_encoding_scannet20)
            image_to_save[:, :self.W, :] = label_pred_image
            # image_to_save[:, 2 * W:3 * W, :] = label_gt_image
        else:
            image_to_save = np.zeros((self.H, self.W * 2, 3))
            # predicted label
            label_pred = output.argmax(0).cpu().numpy()
            label_pred_image = create_label_image(label_pred, color_encoding_scannet20)
            # label_gt_image = create_label_image(label_gt, color_encoding_scannet20)
            # raw image
            img = rgb[0].cpu()  # [3, H, W]
            img = (img * self.PIXEL_STD.view(3, 1, 1) + self.PIXEL_MEAN.view(3, 1, 1)).permute(1, 2, 0).numpy() * 255.  # [H, W, 3]
            image_to_save[:, :self.W, :] = img
            image_to_save[:, self.W:2 * self.W, :] = label_pred_image
            # image_to_save[:, 2 * W:3 * W, :] = label_gt_image
        imageio.imwrite(os.path.join(save_dir, "{}.png".format(frame_id)), image_to_save.astype(np.uint8))


def run_sequential_qpos(exp_dir, scene, test=False, dataset="scannet", mapping_every=20, skip=10):
    """
    Minimal requirement for sequential mapping:
    1. segment.pth: a tensor of [V,] saving per-vertex segment_id
    2. segment.ply (optional): mesh visualizing the segments
    3. valid_segment_mask.pth: a tensor of [N_seg_all,] saving mask for valid segment ids.
    We define invalid segments as segments that contain too few vertices, e.g. <10
    4. knn_mat.pth: a tensor of [N_seg_valid, K] saving the KNN-relationship for every valid segment
    For sequential case, we also need to propagate all those things instead of doing everything from scratch.
    """
    if not test:
        scannet_root = get_scannet_root()
    else:
        scannet_root = get_scannet_test_root()
    scene_dir = os.path.join(scannet_root, scene)
    out_root = os.path.join(exp_dir, dataset)
    output_path = os.path.join(out_root, "{}_skip{}".format(scene, mapping_every))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    path_to_depth = os.path.join(scene_dir, "depth")
    path_to_mesh = os.path.join(scene_dir, "{}_vh_clean_2.ply".format(scene))
    path_to_intri = os.path.join(scene_dir, "intrinsic/intrinsic_depth.txt")

    # segmentation_logger is an example of how to get incremental segmentations.
    # Now it is simply saving them into a ply file for the illustration purposes.
    segmentation_logger = SegmentationLogger(output_path, path_to_mesh, mapping_every=mapping_every)
    process_sequence_with_segmenter(
        scene_dir,
        segmentation_logger,
        path_to_mesh,
        path_to_depth,
        path_to_intri,
        expected_segment_size=0.08,
        small_segment_size=0.01,
        width=640,
        height=480,
        skip=skip)


class BayesianLabelSequential(object):
    """_summary_
    Bayesian label for scannet sequential
    Args:
        object (_type_): _description_
    """
    def __init__(self, exp_dir, scene, window_size=3, model_name="LPN", epoch=19, max_lost=80, H=480, W=640, skip=20, device=torch.device("cuda:0")):
        test = False
        if not test:
            self.scannet_root = get_scannet_root()
        else:
            self.scannet_root = get_scannet_test_root()
        self.device = device
        self.exp_dir = exp_dir
        self.PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406])
        self.PIXEL_STD = torch.tensor([0.229, 0.224, 0.225])

        # load pre-trained model
        self.lpn_logdir = os.path.join(self.exp_dir, "LPN")
        cfg_file = os.path.join(self.lpn_logdir, "config.yaml")
        self.cfg = load_yaml(cfg_file)
        self.cfg.window_size = window_size
        self.model_name = model_name
        self.model = get_model(self.cfg, model_name=self.model_name, device=self.device)
        ckpt_path = os.path.join(self.lpn_logdir, "checkpoints/chkpt-{}.pth".format(epoch))
        pretrained_state_dict = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(pretrained_state_dict["state_dict"])
        self.model.eval()

        # details for the sequence
        self.skip = skip
        self.max_lost = max_lost
        self.window_size = window_size
        self.H = H
        self.W = W

        # Dataset for the scene
        self.scene = scene
        self.scene_segment_root = os.path.join(self.exp_dir, "scannet/{}_skip{}".format(self.scene, self.skip))
        self.data = ScannetScene(scene, test=test, skip=self.skip, H=self.H, W=self.W, load_label=False, load_last=True)
        self.verts = self.data.verts.to(self.device)
        self.faces = self.data.faces.to(self.device)
        self.V, _ = self.verts.shape  # [V,]
        self.vert_probs = torch.ones(self.V, 21, device=device) / 21.  # [V, 21]
        self.timing = {
            "project": [],
            "associate": [],
            "update": []
        }

    @torch.no_grad()
    def run(self):
        # need a queue for storing previous features
        # a frame (frame_id, c2w, w2c, skip1, skip2, bneck_feat)
        mem_q = deque()
        for frame_data in tqdm(self.data):
            # [1, 3, H, W]
            rgb, depth, c2w, w2c, K, frame_id = frame_data["rgb"].unsqueeze(0).to(self.device), \
                frame_data["depth"].unsqueeze(0).to(self.device), \
                frame_data["c2w"], \
                frame_data["w2c"], \
                frame_data["K"].unsqueeze(0).to(self.device), \
                frame_data["frame_id"]

            segment_dir = os.path.join(self.scene_segment_root, "{:06d}".format(frame_id))

            if c2w is not None:
                c2w = c2w.unsqueeze(0).to(self.device)
                w2c = w2c.unsqueeze(0).to(self.device)
                skip1, skip2, feat = self.model.feature_net_forward(rgb, depth_input=depth if self.cfg.use_ssma else None)
                data_tuple = (skip1, skip2, feat, depth, c2w, w2c, frame_id)
                if len(mem_q) > 0 and (frame_id - mem_q[-1][-1]) > self.max_lost:  # lost track, restart
                    mem_q = deque()
                mem_q.append(data_tuple)
                if len(mem_q) > self.window_size:
                    mem_q.popleft()
                out_dict = self.model.causal_forward(mem_q, K, profiler=None)
                class_logit = out_dict["out"][0]
                class_prob = F.softmax(class_logit, dim=0)  # [21, H, W]

                # 2D-3D data association
                frustum_mask = torch.load(os.path.join(segment_dir, "frustum_mask.pth")).to(self.device)
                t1 = get_time()
                verts = self.verts[frustum_mask, :]
                uv_norm, _ = project_pcd(verts, K.squeeze(0), self.H, self.W, depth=None, crop=0, w2c=w2c.squeeze(0), eps=0.05)  # [N_valid, 2]
                t2 = get_time()
                likelihood = F.grid_sample(class_prob.unsqueeze(0), uv_norm.unsqueeze(0).unsqueeze(0), align_corners=False, padding_mode="border").squeeze().t()  # [N_valid, 21]
                t3 = get_time()
                p_post = self.vert_probs[frustum_mask, :] * likelihood
                self.vert_probs[frustum_mask, :] = p_post / torch.sum(p_post, dim=-1, keepdim=True)  # normalize
                t4 = get_time()
                self.timing["project"].append(t2 - t1)
                self.timing["associate"].append(t3 - t2)
                self.timing["update"].append(t4 - t3)

            torch.save(self.vert_probs, os.path.join(segment_dir, "class_prob_bayesian.pth"))
            vert_label = self.vert_probs.argmax(1)
            torch.save(vert_label.long(), os.path.join(segment_dir, "class_label_bayesian.pth"))
            colors = torch.from_numpy(vert_label_to_color(vert_label.cpu().numpy(), color_encoding_scannet20).astype(np.float32) / 255.)
            tex = TexturesVertex(verts_features=colors[None])
            # Only accepts batched input: [B, V, 3], [B, F, 3]
            mesh = Meshes(verts=self.verts[None], faces=self.faces[None], textures=tex)
            # For some reason, colors_as_uint8=True is required to save texture
            IO().save_mesh(mesh, os.path.join(segment_dir, "{:06d}_bayesian.ply".format(frame_id)), colors_as_uint8=True)

        avg_t1 = np.asarray(self.timing["project"]).mean()
        avg_t2 = np.asarray(self.timing["associate"]).mean()
        avg_t3 = np.asarray(self.timing["update"]).mean()
        print("Overall time: {}, Project: {}, Associate: {}, Update: {}".format(avg_t1 + avg_t2 + avg_t3, avg_t1, avg_t2, avg_t3))


class BayesianLabel(object):
    """_summary_
    Bayesian label for scannet offline
    Args:
        object (_type_): _description_
    """
    def __init__(self, model, cfg, log_dir, scene, scene_type, window_size=3, epoch=19, max_lost=80, H=480, W=640, skip=10, device=torch.device("cuda:0")):
        self.scannet_root = get_scannet_root()
        self.device = device
        self.log_dir = log_dir
        self.PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406])
        self.PIXEL_STD = torch.tensor([0.229, 0.224, 0.225])

        # load pre-trained model
        self.model = model
        self.cfg = cfg
        ckpt_path = os.path.join(self.log_dir, "checkpoints/chkpt-{}.pth".format(epoch))
        pretrained_state_dict = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(pretrained_state_dict["state_dict"])
        self.model.eval()

        # details for the sequence
        self.skip = skip
        self.max_lost = max_lost
        self.window_size = window_size
        self.H = H
        self.W = W

        # Dataset for the scene
        self.scene = scene
        self.scene_save_dir = os.path.join(self.log_dir, "label_fusion/scannet/skip_{}/{}_views/bayesian/{}/{}".format(self.skip, window_size, scene_type, scene))
        if not os.path.exists(self.scene_save_dir):
            os.makedirs(self.scene_save_dir)
        self.data = ScannetScene(scene, skip=self.skip, H=self.H, W=self.W, load_label=False, load_last=True)
        self.verts = self.data.verts.to(self.device)
        self.faces = self.data.faces.to(self.device)
        self.V, _ = self.verts.shape  # [V,]
        self.vert_probs = torch.ones(self.V, 21, device=device) / 21.  # [V, 21]
        self.timing = {
            "project": [],
            "associate": [],
            "update": []
        }

    @torch.no_grad()
    def run(self):
        # need a queue for storing previous features
        # a frame (frame_id, c2w, w2c, skip1, skip2, bneck_feat)
        mem_q = deque()
        for frame_data in tqdm(self.data):
            # [1, 3, H, W]
            rgb, depth, c2w, w2c, K, frame_id = frame_data["rgb"].unsqueeze(0).to(self.device), \
                frame_data["depth"].unsqueeze(0).to(self.device), \
                frame_data["c2w"], \
                frame_data["w2c"], \
                frame_data["K"].unsqueeze(0).to(self.device), \
                frame_data["frame_id"]

            if c2w is not None:
                c2w = c2w.unsqueeze(0).to(self.device)
                w2c = w2c.unsqueeze(0).to(self.device)
                skip1, skip2, feat = self.model.feature_net_forward(rgb, depth_input=depth if self.cfg.use_ssma else None)
                data_tuple = (skip1, skip2, feat, depth, c2w, w2c, frame_id)
                if len(mem_q) > 0 and (frame_id - mem_q[-1][-1]) > self.max_lost:  # lost track, restart
                    mem_q = deque()
                mem_q.append(data_tuple)
                if len(mem_q) > self.window_size:
                    mem_q.popleft()
                out_dict = self.model.causal_forward_batch_warp(mem_q, K, profiler=None)
                class_logit = out_dict["out"][0]
                class_prob = F.softmax(class_logit, dim=0)  # [21, H, W]

                # 2D-3D data association
                t1 = get_time()
                verts = self.verts
                # print(depth.shape)
                uv_norm, valid_mask = project_pcd(verts, K.squeeze(0), self.H, self.W, depth=depth.squeeze(0).squeeze(0), crop=0, w2c=w2c.squeeze(0), eps=0.05)  # [N_valid, 2]
                valid_uv_norm = uv_norm[valid_mask, :]
                t2 = get_time()
                likelihood = F.grid_sample(class_prob.unsqueeze(0), valid_uv_norm.unsqueeze(0).unsqueeze(0), align_corners=False, padding_mode="border").squeeze().t()  # [N_valid, 21]
                t3 = get_time()
                p_post = self.vert_probs[valid_mask, :] * likelihood
                self.vert_probs[valid_mask, :] = p_post / torch.sum(p_post, dim=-1, keepdim=True)  # normalize
                t4 = get_time()
                self.timing["project"].append(t2 - t1)
                self.timing["associate"].append(t3 - t2)
                self.timing["update"].append(t4 - t3)

        torch.save(self.vert_probs, os.path.join(self.scene_save_dir, "class_prob_bayesian.pth"))
        vert_label = self.vert_probs.argmax(1)
        torch.save(vert_label.long(), os.path.join(self.scene_save_dir, "class_label_bayesian.pth"))
        colors = torch.from_numpy(vert_label_to_color(vert_label.cpu().numpy(), color_encoding_scannet20).astype(np.float32) / 255.)
        tex = TexturesVertex(verts_features=colors[None])
        # Only accepts batched input: [B, V, 3], [B, F, 3]
        mesh = Meshes(verts=self.verts[None], faces=self.faces[None], textures=tex)
        # For some reason, colors_as_uint8=True is required to save texture
        IO().save_mesh(mesh, os.path.join(self.scene_save_dir, "labelled_mesh_{}_bayesian.ply".format(self.scene)), colors_as_uint8=True)

        avg_t1 = np.asarray(self.timing["project"]).mean()
        avg_t2 = np.asarray(self.timing["associate"]).mean()
        avg_t3 = np.asarray(self.timing["update"]).mean()
        print("Overall time: {}, Project: {}, Associate: {}, Update: {}".format(avg_t1 + avg_t2 + avg_t3, avg_t1, avg_t2, avg_t3))


# TODO: 20231107 Where to put these two online dataset??? Merge this into dataio.scannet
class ScannetScene(torch.utils.data.Dataset):
    def __init__(self, scene, test=False, adaptive=False, min_angle=15.0, min_distance=0.1, skip=20,
                 load_all=True, load_label=True, load_last=False, H=480, W=640, seg_classes="scannet20"):
        if not test:
            self.scannet_root = get_scannet_root()
        else:
            self.scannet_root = get_scannet_test_root()
        # 3D mesh verts
        self.mesh_file = os.path.join(self.scannet_root, scene, "{}_vh_clean_2.ply".format(scene))
        ply_dict = get_mesh_vt(self.mesh_file)
        verts_np, faces_np = ply_dict["verts"], ply_dict["faces"]
        self.verts = torch.from_numpy(verts_np)
        self.faces = torch.from_numpy(faces_np)

        self.skip = skip
        self.H = H
        self.W = W
        self.seg_classes = seg_classes
        if self.seg_classes.lower() == "nyu40":
            self.num_classes = 41
        elif self.seg_classes.lower() == "scannet20":
            self.num_classes = 21
        else:
            raise NotImplementedError
        self.color_encoding = self.get_color_encoding()
        self.transforms = get_transforms(height=self.H, width=self.W,
                                         height_test=None,
                                         width_test=None,
                                         phase="test")
        self.load_all = load_all
        self.load_last = load_last
        self.load_label = load_label
        self.scene = scene
        # get paths to all the images, depths and labels
        self.rgb_paths = []
        self.depth_paths = []
        self.label_paths = []
        self.frame_ids = []
        self.c2w_list = []
        self.w2c_list = []

        scene_dir = os.path.join(self.scannet_root, self.scene)
        rgb_dir = os.path.join(scene_dir, "color")
        depth_dir = os.path.join(scene_dir, "depth")
        label_dir = os.path.join(scene_dir, "label-{}".format(self.H))
        pose_dir = os.path.join(scene_dir, "pose")
        intrinsic_dir = os.path.join(scene_dir, "intrinsic")
        self.K = np.loadtxt(os.path.join(intrinsic_dir, "intrinsic_depth.txt"))[:3, :3].astype(np.float32)

        scene_len = len(os.listdir(pose_dir))
        last_id = scene_len - 1
        self.adaptive = adaptive
        if self.adaptive:
            self.frame_list = []
            last_pose = None
            for i in range(len(os.listdir(self.pose_dir))):
                c2w = torch.from_numpy(np.loadtxt(os.path.join(self.pose_dir, "{:d}.txt".format(i))).astype(np.float32).reshape(4, 4))
                if torch.isnan(c2w).any().item() or torch.isinf(c2w).any().item():
                    continue
                if last_pose is None:  # first frame
                    self.frame_list.append(i)
                    last_pose = c2w
                    continue
                unit_vec = torch.tensor([[0.], [0.], [1.]])
                angle = torch.acos(((c2w[:3, :3].t() @ last_pose[:3, :3] @ unit_vec) * unit_vec).sum())
                distance = torch.norm(c2w[:3, 3] - last_pose[:3, 3])

                if angle > (min_angle / 180) * np.pi or distance > min_distance:  # create a new keyframe
                    self.frame_list.append(i)
                    last_pose = c2w
        else:
            self.frame_list = list(range(0, scene_len, self.skip))
            if self.load_last and last_id not in self.frame_list:
                self.frame_list.append(last_id)
        for i in self.frame_list:
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
            if np.isnan(c2w).any() or np.isinf(c2w).any():
                self.c2w_list.append(None)
                self.w2c_list.append(None)
            else:
                self.c2w_list.append(c2w)
                self.w2c_list.append(np.linalg.inv(c2w))
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
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)

        depth = np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0
        H_orig, W_orig = depth.shape
        s = float(H_orig) / float(self.H)
        K_depth = copy.deepcopy(self.K)
        K_depth[0, :] /= s
        K_depth[1, :] /= s
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)  # [H, W]

        sample = {
            "rgb": rgb,
            "depth": depth,
            "frame_id": self.frame_ids[idx]
        }

        if self.load_all:
            sample["K"] = K_depth
            sample["c2w"] = self.c2w_list[idx]
            sample["w2c"] = self.w2c_list[idx]

        if self.load_label:
            label_path = self.label_paths[idx]
            label = np.array(imageio.imread(label_path))
            if self.seg_classes == "scannet20":
                label = nyu40_to_scannet20(label)
            # in case there are some invalid labels, but this shouldn't happen?
            label[label > 20] = 0
            sample["label"] = label

        sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.rgb_paths)
    

class ScannetSceneSegments(torch.utils.data.Dataset):
    def __init__(self, exp_dir, scene, dataset="scannet", skip=20, k=64):
        self.dataset_type = dataset
        self.scene_dir = os.path.join(exp_dir, "{}/{}_skip{}".format(dataset, scene, skip))
        frame_list = sorted(os.listdir(self.scene_dir))
        # Need to construct K-NN on-the-fly
        self.seg_center_list = []
        self.seg_cov_list = []
        # self.seg_label_list = []
        self.seg_feat_list = []
        self.knn_mat_list = []
        self.frame_id_list = []
        for frame in frame_list:
            frame_dir = os.path.join(self.scene_dir, frame)
            self.seg_center_list.append(torch.load(os.path.join(frame_dir, "seg_center.pth")).float())  # [N_seg, 3]
            self.seg_cov_list.append(torch.load(os.path.join(frame_dir, "seg_cov.pth")).float())  # [N_seg, 3, 3]
            # input feature
            feats = torch.load(os.path.join(frame_dir, "seg_feat_prob.pth")).float()  # [N_seg, C]
            self.seg_feat_list.append(feats)
            # knn_mat
            knn_mat = torch.load(os.path.join(frame_dir, "nn_mat.pth")).long()[:, :k]  # [N_seg, K]
            self.knn_mat_list.append(knn_mat)
            self.frame_id_list.append(int(frame))

    def __getitem__(self, idx):
        # return data point
        data = {
            "locs": self.seg_center_list[idx],
            "covs": self.seg_cov_list[idx],
            "feats": self.seg_feat_list[idx],
            "knn_indices": self.knn_mat_list[idx],
            "frame_id": self.frame_id_list[idx]
        }
        return data

    def __len__(self):
        # length of the sequence
        return len(self.seg_center_list)


class Profiler(object):
    def __init__(self) -> None:
        self.encoder_timing = []
        self.feature_warp_timing1 = []
        self.feature_warp_timing2 = []
        self.feature_warp_timing3 = []
        self.feature_fusion_timing = []
        self.decoder_timing = []
    
    def append_encoder(self, t):
        self.encoder_timing.append(t)
    
    def append_feature_skip1(self, t):
        self.feature_warp_timing1.append(t)
    
    def append_feature_skip2(self, t):
        self.feature_warp_timing2.append(t)
    
    def append_feature_bneck(self, t):
        self.feature_warp_timing3.append(t)
        
    def append_decoder(self, t):
        self.decoder_timing.append(t)
        
    def append_feature_fusion(self, t):
        self.feature_fusion_timing.append(t)
        
    def __len__(self):
        return len(self.encoder_timing)
        
    def log(self):
        avg_t1 = np.asarray(self.encoder_timing).mean()
        avg_t2 = np.asarray(self.feature_warp_timing1).mean()
        avg_t3 = np.asarray(self.feature_warp_timing2).mean()
        avg_t4 = np.asarray(self.feature_warp_timing3).mean()
        avg_t5 = np.asarray(self.feature_fusion_timing).mean()
        avg_t6 = np.asarray(self.decoder_timing).mean()
        print(self.__len__())
        print("Average encoder timing: {}".format(avg_t1))
        print("Average feature re-projection timing: {} = {} + {} + {}".format(avg_t2 + avg_t3 + avg_t4, avg_t2, avg_t3, avg_t4))
        print("Average feature fusion timing: {}".format(avg_t5))
        print("Average decoder timing: {}".format(avg_t6))


def get_mesh_vt(mesh_file):
    """
    :param mesh_file: must be "{}_vh_clean_2.labels.ply"
    :return:
    """
    label_ply = plyfile.PlyData().read(mesh_file)
    verts = np.stack([np.asarray(label_ply.elements[0]["x"]),
                      np.asarray(label_ply.elements[0]["y"]),
                      np.asarray(label_ply.elements[0]["z"])], axis=1).astype(np.float32)
    colors = np.stack([np.asarray(label_ply.elements[0]["red"]),
                       np.asarray(label_ply.elements[0]["green"]),
                       np.asarray(label_ply.elements[0]["blue"])], axis=1).astype(np.float32) / 255.
    # normals = np.stack([np.asarray(label_ply.elements[0]["Nx"]),
    #                     np.asarray(label_ply.elements[0]["Ny"]),
    #                     np.asarray(label_ply.elements[0]["Nz"])], axis=1).astype(np.float32)
    faces = np.stack(label_ply.elements[1]["vertex_indices"], axis=0).astype(np.int64)

    ret = {
        "verts": verts,
        "faces": faces,
        # "normals": normals,
        "colors": colors
    }
    return ret


def create_3d_data_for_sequential_experiment(exp_dir, scene, test=False, dataset="scannet", skip=20, th_bad_seg=10, k=128):
    """
    :param exp_dir:
    :param scene:
    :param test:
    :param dataset:
    :param skip:
    :param th_bad_seg:
    :param k:
    :return:
    """
    if not test:
        scannet_root = get_scannet_root()
    else:
        scannet_root = get_scannet_test_root()
    ply_dict = get_mesh_vt(os.path.join(os.path.join(scannet_root, scene, "{}_vh_clean_2.ply".format(scene))))
    verts, faces, colors = ply_dict["verts"], ply_dict["faces"], ply_dict["colors"]
    if os.path.exists(os.path.join(scannet_root, scene, "mesh_normals.pth")):
        normals = torch.load(os.path.join(scannet_root, scene, "mesh_normals.pth")).cpu().numpy()
    else:
        normals = compute_normals_o3d(verts, faces)
        torch.save(torch.from_numpy(normals), os.path.join(scannet_root, scene, "mesh_normals.pth"))
    scene_dir = os.path.join(exp_dir, "{}/{}_skip{}".format(dataset, scene, skip))
    frames = sorted(os.listdir(scene_dir))
    T1, T2 = [], []
    for frame in tqdm(frames):
        frame_dir = os.path.join(scene_dir, frame)
        segments = torch.load(os.path.join(frame_dir, "segments.pth")).cpu().numpy()
        class_prob = torch.load(os.path.join(frame_dir, "class_prob_bayesian.pth")).cpu().numpy()
        n_classes = class_prob.shape[1]
        # count -1 as invalid segment as well
        seg_ids_all = np.unique(segments)
        N_seg_all = len(seg_ids_all)
        valid_segments_mask = np.zeros((N_seg_all,)).astype(bool)  # [N_seg_all,]
        seg_center = np.zeros((N_seg_all, 3))  # [N_seg_all, 3]
        seg_cov = np.zeros((N_seg_all, 3, 3))  # [N_seg_all, 3, 3]
        seg_feat_prob = np.zeros((N_seg_all, n_classes + 9))  # [N_seg_all, C]

        t1 = get_time()
        for i, seg_id in enumerate(seg_ids_all):
            seg_mask = (segments == seg_id)
            if seg_id == -1 or len(verts[seg_mask, :]) < th_bad_seg:
                valid_segments_mask[i] = False
                continue

            valid_segments_mask[i] = True
            # location
            seg_center[i] = np.mean(verts[seg_mask, :], axis=0)
            # cov-matrix
            seg_cov[i] = np.cov(verts[seg_mask, :].transpose())
            # normal
            mean_normals = np.mean(normals[seg_mask, :], axis=0)  # [3,]
            mean_normals = mean_normals / np.linalg.norm(mean_normals)
            # center, color, prob
            mean_colors = np.mean(colors[seg_mask, :], axis=0)  # [3,]
            mean_centers = np.mean(verts[seg_mask, :], axis=0)  # [3,]
            mean_probs = np.mean(class_prob[seg_mask, :], axis=0)  # [21,]
            # feature_prob
            seg_feat_prob[i, :3] = mean_colors
            seg_feat_prob[i, 3:6] = mean_normals
            seg_feat_prob[i, 6:9] = mean_centers
            seg_feat_prob[i, 9:] = mean_probs

        # valid_segment exists
        if np.count_nonzero(~valid_segments_mask) > 0:
            # get valid segments only
            N_seg = np.count_nonzero(valid_segments_mask)
            seg_center = seg_center[valid_segments_mask, :]
            seg_cov = seg_cov[valid_segments_mask, :, :]
            seg_feat_prob = seg_feat_prob[valid_segments_mask, :]
        else:
            N_seg = N_seg_all

        t2 = get_time()
        T1.append(t2- t1)
        # save valid_segments_mask and KNN matrix
        torch.save(torch.from_numpy(valid_segments_mask), os.path.join(frame_dir, "valid_segments_mask.pth"))  # [N_seg_all]
        # create KD-tree
        t3 = get_time()
        tree = KDTree(seg_center)
        # Store NN-matrix instead of hard-coded K-NN features
        nn_mat = tree.query(seg_center, k=k, return_distance=False, sort_results=True)  # [N_seg, N_seg]
        t4 = get_time()
        T2.append(t4 - t3)
        torch.save(torch.from_numpy(nn_mat), os.path.join(frame_dir, "nn_mat.pth"))  # [N_seg, N_seg]
        # save segment features
        torch.save(torch.from_numpy(seg_center), os.path.join(frame_dir, "seg_center.pth"))  # [N_seg, 3]
        torch.save(torch.from_numpy(seg_cov), os.path.join(frame_dir, "seg_cov.pth"))  # [N_seg, 3, 3]
        torch.save(torch.from_numpy(seg_feat_prob), os.path.join(frame_dir, "seg_feat_prob.pth"))  # [N_seg, C]


def apply_3dconv_for_sequental_experiment(exp_dir, scene, test=False, dataset="scannet", skip=20, k=64, epoch="best", device=torch.device("cuda:0")):
    if not test:
        scannet_root = get_scannet_root()
    else:
        scannet_root = get_scannet_test_root()
    scene_data_dir = os.path.join(scannet_root, scene)
    log_dir = os.path.join(exp_dir, "SegConvNet")
    cfg_file = os.path.join(log_dir, "config.yaml")
    cfg = load_yaml(cfg_file)
    model = get_model_3d(cfg)
    model.to(device)
    chkpt = torch.load(os.path.join(log_dir, "checkpoints/chkpt-{}.pth".format(epoch)), map_location=device)
    model.load_state_dict(chkpt["state_dict"])
    model.eval()
    scene_data = ScannetSceneSegments(exp_dir, scene, dataset=dataset, skip=skip, k=k)
    T = []
    for frame_data in tqdm(scene_data):
        xyz = frame_data["locs"].unsqueeze(0).to(device)  # [1, N_seg, 3]
        cov = frame_data["covs"].unsqueeze(0).to(device)  # [1, N_seg, 3, 3]
        feat = frame_data["feats"].unsqueeze(0).to(device)  # [1, N_seg, C]
        knn_indices = frame_data["knn_indices"].to(device)  # [N_seg, K]
        frame_id = frame_data["frame_id"]

        # Forward propagation
        t1 = get_time()
        out = model(xyz, cov, feat, knn_indices).squeeze()  # [n_classes, N_seg]
        seg_label_pred = torch.argmax(out, dim=0).cpu().numpy()  # [N_seg,]
        t2 = get_time()
        T.append(t2 - t1)

        # load segment.pth
        frame_dir = os.path.join(exp_dir, dataset, "{}_skip{}/{:06d}".format(scene, skip, frame_id))
        segments = torch.load(os.path.join(frame_dir, "segments.pth")).cpu().numpy()  # [V,] with N_seg_all ids
        # initialize with bayesian labelled results
        verts_label_pred = torch.load(os.path.join(frame_dir, "class_label_bayesian.pth")).cpu().numpy()  # [V,]
        assert len(segments) == len(verts_label_pred), "Number of vertices mismatch!!!"
        seg_ids_all = np.unique(segments)  # [N_seg_all,]
        if os.path.exists(os.path.join(frame_dir, "valid_segments_mask.pth")):
            valid_segments_mask = torch.load(os.path.join(frame_dir, "valid_segments_mask.pth")).cpu().numpy()  # [N_seg_all,]
            seg_ids = seg_ids_all[valid_segments_mask]  # [N_seg]
        else:
            seg_ids = seg_ids_all
        assert len(seg_label_pred) == len(seg_ids), "Number of segments mismatch!!!"
        N_seg = len(seg_label_pred)

        # This doesn't seem necessary
        # if N_seg != (seg_ids.max() + 1):
        #     print(frame_id)

        # 0 to N_seg - 1
        for i, seg_id in enumerate(seg_ids):
            # in very rare cases, the seg_ids is not range(N_seg), but still don't know why...
            seg_mask = (segments == seg_id)
            verts_label_pred[seg_mask] = seg_label_pred[i]
        save_dir = os.path.join(frame_dir, "final_results")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(torch.from_numpy(verts_label_pred), os.path.join(save_dir, "verts_label.pth"))
        # save labelled mesh
        ply_dict = get_mesh_vt(os.path.join(scene_data_dir, "{}_vh_clean_2.ply".format(scene)))
        verts, faces = ply_dict["verts"], ply_dict["faces"]
        # mesh_ref = trimesh.load(os.path.join(frame_dir, "{:06d}_bayesian.ply".format(frame_id)))
        # verts, faces = mesh_ref.vertices, mesh_ref.faces
        vert_colors = vert_label_to_color(verts_label_pred, color_encoding_scannet20)
        assert verts.shape[0] == verts_label_pred.shape[0], "Number of vertices doesn't match!!!"
        mesh_save = trimesh.Trimesh(verts, faces, vertex_colors=vert_colors)
        mesh_save.export(os.path.join(save_dir, "verts_label.ply"))
        obs_mask = torch.load(os.path.join(frame_dir, "obs_mask.pth")).cpu().numpy()
        # obs_mask &= (segments != -1)
        mesh_save_clean = clean_mesh(mesh_save, obs_mask)
        mesh_save_clean.export(os.path.join(save_dir, "verts_label_clean.ply"))


def clean_mesh(mesh, mask):
    verts, faces, colors = mesh.vertices, mesh.faces, mesh.visual.vertex_colors
    valid_faces_mask = (mask[faces[:, 0]] | mask[faces[:, 1]] | mask[faces[:, 2]])
    valid_faces = faces[valid_faces_mask, :]
    cleaned_mesh = trimesh.Trimesh(verts, valid_faces, vertex_colors=colors)
    cleaned_mesh.remove_unreferenced_vertices()
    return cleaned_mesh


def get_model(cfg, model_name="LPN", num_classes=21, pretrained=True, device=torch.device("cuda")):
    if model_name == "LPN":
        Model2D = LatentPriorNetwork
    else:
        Model2D = MVCNet
    model = Model2D(num_classes=num_classes,
                    pretrained=pretrained,
                    modality=cfg.setdefault("modality", "rgbd"),
                    use_ssma=cfg.setdefault("use_ssma", True),
                    reproject=cfg.setdefault("reproject", True),
                    decoder_in_dim=cfg.decoder_in_dim,
                    decoder_feat_dim=cfg.decoder_feat_dim,
                    decoder_head_dim=cfg.decoder_head_dim,
                    window_size=cfg.setdefault("window_size", 1),
                    projection_dims=cfg.projection_dim,
                    render=cfg.setdefault("render", False),
                    fusion_mode=cfg.setdefault("fusion_mode", "average")).to(device)
    return model


def get_model_3d(cfg):
    in_dim = 30 if cfg.use_xyz else 21
    model = SegConvNet(input_feat_dim=in_dim,
                       num_classes=21,
                       dropout_p=cfg.dropout_p,
                       weight_in=cfg.setdefault("weight_in", "xyz"),
                       classifier_hidden_dims=cfg.setdefault("classifier_hidden_dims", [128, 64]))
    return model


def inference_one_scene(logdir, scene, mode="causal", window_size=3, skip_invalid=False, save=True, load_label=False):
    assert mode in ["causal", "causal_fused"]
    runner = ScannetInferenceRunner(logdir, window_size=window_size)
    if mode == "causal":
        runner.run_causal_inference(scene, save=save, skip_invalid=skip_invalid, load_label=load_label)
    else:
        runner.run_causal_inference_fused(scene, save=save, skip_invalid=skip_invalid, load_label=load_label)


def eval(logdir, mode="causal", window_size=5, skip_invalid=False, save=False, load_label=True):
    assert mode in ["causal", "causal_fused"]
    runner = ScannetInferenceRunner(logdir, window_size=window_size)
    metric = IoU(num_classes=21, ignore_index=0)
    scenes = get_scene_list("configs/scannetv2_val.txt")
    for i, scene in enumerate(tqdm(scenes)):
        if mode == "causal":
            runner.run_causal_inference(scene, metric=metric, save=save, skip_invalid=skip_invalid, load_label=load_label)
        else:
            runner.run_causal_inference_fused(scene, metric=metric, save=save, skip_invalid=skip_invalid, load_label=load_label)
        # runner.run_batched_inference(scene, metric=metric, save=save, skip_invalid=skip_invalid)
        # runner.run_causal_inference_fused(scene, metric=metric, save=True, skip_invalid=False)
        runner.profiler.log()

    iou, miou = metric.value()
    with open(os.path.join(logdir, "evaluate_{}_{}_invalid_{}views.txt".format(mode, "skip" if skip_invalid else "not_skip", runner.window_size)), "w") as f:
        f.write("-------------Evaluation Result--------------\n")
        for key, class_iou in zip(color_encoding_scannet20.keys(), iou):
            f.write("{0}: {1:.4f}\n".format(key, class_iou))
        f.write("Mean IoU: {}\n\n".format(miou))
    
    np.savetxt(os.path.join(logdir, "{}_{}_invalid_{}views_conf_mat.txt".format(mode, "skip" if skip_invalid else "not_skip", runner.window_size)), metric.conf_metric.conf, fmt="%10d")
    print("Processed {} images".format(metric.counter))
    
    
def eval_causal_batch_warp(logdir, mode="causal", window_size=5, skip_invalid=False, save=False, load_label=True):
    assert mode in ["causal", "causal_fused"]
    runner = ScannetInferenceRunner(logdir, window_size=window_size)
    # scene = "scene0191_00"
    # runner.run_causal_inference(scene, save=True)
    # runner.run_causal_inference_fused(scene, save=True)
    metric = IoU(num_classes=21, ignore_index=0)
    scenes = get_scene_list("configs/scannetv2_val.txt")
    for i, scene in enumerate(tqdm(scenes)):
        if mode == "causal":
            runner.run_causal_inference_batch_warp(scene, metric=metric, save=save, skip_invalid=skip_invalid, load_label=load_label)
        else:
            runner.run_causal_inference_fused(scene, metric=metric, save=save, skip_invalid=skip_invalid, load_label=load_label)
        # runner.run_batched_inference(scene, metric=metric, save=save, skip_invalid=skip_invalid)
        # runner.run_causal_inference_fused(scene, metric=metric, save=True, skip_invalid=False)
        runner.profiler.log()

    iou, miou = metric.value()
    with open(os.path.join(logdir, "evaluate_{}_batch_warp_{}_invalid_{}views.txt".format(mode, "skip" if skip_invalid else "not_skip", runner.window_size)), "w") as f:
        f.write("-------------Evaluation Result--------------\n")
        for key, class_iou in zip(color_encoding_scannet20.keys(), iou):
            f.write("{0}: {1:.4f}\n".format(key, class_iou))
        f.write("Mean IoU: {}\n\n".format(miou))
    
    np.savetxt(os.path.join(logdir, "{}_batch_warp_{}_invalid_{}views_conf_mat.txt".format(mode, "skip" if skip_invalid else "not_skip", runner.window_size)), metric.conf_metric.conf, fmt="%10d")
    print("Processed {} images".format(metric.counter))


def eval_batched(logdir, window_size=5, skip_invalid=False, save=False):
    runner = ScannetInferenceRunner(logdir, window_size=window_size)
    # scene = "scene0191_00"
    # runner.run_causal_inference(scene, save=True)
    # runner.run_causal_inference_fused(scene, save=True)
    metric = IoU(num_classes=21, ignore_index=0)
    scenes = get_scene_list("configs/scannetv2_val.txt")
    for scene in tqdm(scenes):
        # runner.run_causal_inference(scene, metric=metric, save=save, skip_invalid=skip_invalid)
        runner.run_batched_inference(scene, metric=metric, save=save, skip_invalid=skip_invalid)
        # runner.run_causal_inference_fused(scene, metric=metric, save=True, skip_invalid=False)

    iou, miou = metric.value()
    with open(os.path.join(logdir, "evaluate_batch_{}_invalid_{}views.txt".format("skip" if skip_invalid else "not_skip", runner.window_size)), "w") as f:
        f.write("-------------Evaluation Result--------------\n")
        for key, class_iou in zip(color_encoding_scannet20.keys(), iou):
            f.write("{0}: {1:.4f}\n".format(key, class_iou))
        f.write("Mean IoU: {}\n\n".format(miou))
    
    np.savetxt(os.path.join(logdir, "batch_{}_invalid_{}views_conf_mat.txt".format("skip" if skip_invalid else "not_skip", runner.window_size)), metric.conf_metric.conf, fmt="%10d")
    print("Processed {} images".format(metric.counter))


def run_scannet_bayesian_labelling(logdir, window_size=5):
    cfg_file = os.path.join(logdir, "config.yaml")
    cfg = load_yaml(cfg_file)
    cfg.window_size = window_size
    model = get_model(cfg, model_name="LPN", device=torch.device("cuda:0"))
    scene_list_train = get_scene_list("configs/scannetv2_train.txt")
    
    for scene in tqdm(scene_list_train):
        runner = BayesianLabel(model, cfg, logdir, scene, "train", window_size=window_size, epoch=19, max_lost=80)
        runner.run()
        
    scene_list_val = get_scene_list("configs/scannetv2_val.txt")
    for scene in tqdm(scene_list_val):
        runner = BayesianLabel(model, cfg, logdir, scene, "val", window_size=window_size, epoch=19, max_lost=80)
        runner.run()


def run_sequential_mapping():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--scene", type=str, default="scene0645_00")
    parser.add_argument("--model_type", type=str, default="LPN_rgb")
    parser.add_argument("--mapping_every", type=int, default=20)
    parser.add_argument("--skip", type=int, default=1, help="dataset skip")
    args = parser.parse_args()
    exp_dir = args.exp_dir
    scene = args.scene

    run_sequential_qpos(exp_dir, scene, mapping_every=args.mapping_every, skip=args.skip)
    bayesian_runner = BayesianLabelSequential(exp_dir, scene)
    bayesian_runner.run()
    create_3d_data_for_sequential_experiment(exp_dir, scene)
    apply_3dconv_for_sequental_experiment(exp_dir, scene)


if __name__ == "__main__":
    run_sequential_mapping()
