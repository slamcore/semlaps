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
import argparse
import shutil
from collections import deque
import time

import numpy as np
import torch
import torch.nn.functional as F
import plyfile
import cv2
import imageio
from tqdm import tqdm
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
)
import trimesh
import open3d as o3d
from sklearn.neighbors import KDTree

from networks.LatentPriorNetwork import LatentPriorNetwork
from networks.SegConvNet import SegConvNet
from dataio.transforms_base import get_transforms
from dataio.utils import color_encoding_nyu40, vert_label_to_color, color_encoding_scannetX
from networks.rend_utils import project_pcd
from config import get_slamcore_root, load_yaml
from qpos.segment_mesh_online_v3 import SegmentationLogger, process_sequence_with_segmenter
from dataio.utils import create_label_image, color_encoding_scannet20
from train_lpn import get_model as get_lpn_model
from train_segconvnet import get_model as get_segconv_model

def get_mesh_vt(mesh_file):
    """
    :param mesh_file: must be "{}_vh_clean_2.labels.ply"
    :return:
    """
    label_ply = plyfile.PlyData().read(mesh_file)
    verts = np.stack([np.asarray(label_ply.elements[0]["x"]),
                      np.asarray(label_ply.elements[0]["y"]),
                      np.asarray(label_ply.elements[0]["z"])], axis=1).astype(np.float32)
    faces = np.stack(
        label_ply.elements[1]["vertex_index"], axis=0).astype(np.int64)
    segments = np.asarray(label_ply.elements[0]["instance"]).astype(np.int32)
    colors = np.stack([np.asarray(label_ply.elements[0]["red"]),
                       np.asarray(label_ply.elements[0]["green"]),
                       np.asarray(label_ply.elements[0]["blue"])], axis=1).astype(np.float32) / 255.
    ret = {
        "verts": verts,
        "faces": faces,
        "colors": colors,
        "segments": segments,
    }
    return ret


def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()


def run_sequential_qpos(exp_dir, scene, dataset="slamcore", mapping_every=20, skip=10, depth_diff_threshold=0.25):
    """
    Minimal requirement for sequential mapping:
    1. segment.pth: a tensor of [V,] saving per-vertex segment_id
    2. segment.ply (optional): mesh visualizing the segments
    3. valid_segment_mask.pth: a tensor of [N_seg_all,] saving mask for valid segment ids.
    We define invalid segments as segments that contain too few vertices, e.g. <10
    4. knn_mat.pth: a tensor of [N_seg_valid, K] saving the KNN-relationship for every valid segment
    For sequential case, we also need to propagate all those things instead of doing everything from scratch.
    """

    scene_data_root = get_slamcore_root()
    scene_dir = os.path.join(scene_data_root, scene)
    out_root = os.path.join(exp_dir, dataset)
    output_path = os.path.join(out_root, "{}_skip{}".format(scene, mapping_every))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path_to_depth = os.path.join(scene_dir, "depth")
    path_to_mesh = os.path.join(scene_dir, "global_map_mesh.clean.ply")
    path_to_intri = os.path.join(scene_dir, "K.txt")

    # segmentation_logger is an example of how to get incremental segmentations.
    # Now it is simply saving them into a ply file for the illustration purposes.
    segmentation_logger = SegmentationLogger(
        output_path, path_to_mesh, mapping_every=mapping_every)
    process_sequence_with_segmenter(
        scene_dir,
        segmentation_logger,
        path_to_mesh,
        path_to_depth,
        path_to_intri,
        max_num_vertices=240,
        small_segment_size=120,
        width=848,
        height=480,
        skip=skip,
        depth_diff_threshold=depth_diff_threshold)


class BayesianLabelSequential(object):
    def __init__(self, exp_dir, scene, model_type="LPN_rgb", window_size=3, epoch=19, max_lost=80, H=480, W=848, skip=20, center_crop=False, device=torch.device("cuda:0")):
        self.scannet_root = get_slamcore_root()
        self.device = device
        self.exp_dir = exp_dir
        self.PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406])
        self.PIXEL_STD = torch.tensor([0.229, 0.224, 0.225])

        # load pre-trained model
        self.lpn_logdir = os.path.join(self.exp_dir, model_type)
        cfg_file = os.path.join(self.lpn_logdir, "config.yaml")
        self.cfg = load_yaml(cfg_file)
        self.cfg.window_size = window_size
        self.model = get_lpn_model(self.cfg, device=self.device)
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
        self.scene_segment_root = os.path.join(
            self.exp_dir, "slamcore/{}_skip{}".format(self.scene, self.skip))
        self.data = SLAMcoreScene(scene, skip=self.skip, H=self.H, W=self.W, center_crop=center_crop, load_last=True)
        self.W = self.data.W_crop
        self.verts = self.data.verts.to(self.device)
        self.faces = self.data.faces.to(self.device)
        self.V, _ = self.verts.shape  # [V,]
        self.vert_probs = torch.ones(
            self.V, 21, device=device) / 21.  # [V, 21]
        self.timing = {
            "project": [],
            "associate": [],
            "update": []
        }

        self.save2DLabels = False

    @torch.no_grad()
    def run(self, eps=0.20):
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

            segment_dir = os.path.join(
                self.scene_segment_root, "{:06d}".format(frame_id))

            if c2w is not None:
                c2w = c2w.unsqueeze(0).to(self.device)
                w2c = w2c.unsqueeze(0).to(self.device)
                skip1, skip2, feat = self.model.feature_net_forward(rgb, None)
                data_tuple = (skip1, skip2, feat, depth, c2w, w2c, frame_id)
                # lost track, restart
                if len(mem_q) > 0 and (frame_id - mem_q[-1][-1]) > self.max_lost:
                    mem_q = deque()
                mem_q.append(data_tuple)
                if len(mem_q) > self.window_size:
                    mem_q.popleft()
                out_dict = self.model.causal_forward(mem_q, K, profiler=None)
                class_logit = out_dict["out"][0]
                class_prob = F.softmax(class_logit, dim=0)  # [21, H, W]

                if self.save2DLabels:
                    # temporary debug
                    label_pred = class_prob.argmax(0).detach().cpu()
                    label_pred = label_pred.reshape(self.H, self.W)
                    label_pred_image = create_label_image(
                        label_pred, color_encoding_scannet20)
                    os.makedirs(self.scene_segment_root +
                                "/labels2D", exist_ok=True)
                    save_path = os.path.join(
                        self.scene_segment_root, "labels2D/{:06d}.png".format(frame_id))
                    imageio.imwrite(save_path, label_pred_image)

                # 2D-3D data association
                frustum_mask = torch.load(os.path.join(segment_dir, "frustum_mask.pth")).to(self.device)
                t1 = get_time()
                verts = self.verts[frustum_mask, :]
                uv_norm, _ = project_pcd(verts, K.squeeze(0), self.H, self.W, depth=None, crop=0, w2c=w2c.squeeze(0), eps=eps)  # [N_valid, 2]
                t2 = get_time()
                likelihood = F.grid_sample(class_prob.unsqueeze(0), uv_norm.unsqueeze(0).unsqueeze(0), align_corners=False, padding_mode="border").squeeze().t()  # [N_valid, 21]
                t3 = get_time()
                p_post = self.vert_probs[frustum_mask, :] * likelihood
                self.vert_probs[frustum_mask, :] = p_post / \
                    torch.sum(p_post, dim=-1, keepdim=True)  # normalize
                t4 = get_time()
                self.timing["project"].append(t2 - t1)
                self.timing["associate"].append(t3 - t2)
                self.timing["update"].append(t4 - t3)

            torch.save(self.vert_probs, os.path.join(
                segment_dir, "class_prob_bayesian.pth"))
            vert_label = self.vert_probs.argmax(1)
            torch.save(vert_label.long(), os.path.join(
                segment_dir, "class_label_bayesian.pth"))
            colors = torch.from_numpy(vert_label_to_color(
                vert_label.cpu().numpy(), color_encoding_scannet20).astype(np.float32) / 255.)
            tex = TexturesVertex(verts_features=colors[None])
            # Only accepts batched input: [B, V, 3], [B, F, 3]
            mesh = Meshes(verts=self.verts[None],
                          faces=self.faces[None], textures=tex)
            # For some reason, colors_as_uint8=True is required to save texture
            IO().save_mesh(mesh, os.path.join(segment_dir, "{:06d}_bayesian.ply".format(frame_id)), colors_as_uint8=True)

        avg_t1 = np.asarray(self.timing["project"]).mean()
        avg_t2 = np.asarray(self.timing["associate"]).mean()
        avg_t3 = np.asarray(self.timing["update"]).mean()
        print("Overall time: {}, Project: {}, Associate: {}, Update: {}".format(avg_t1 + avg_t2 + avg_t3, avg_t1, avg_t2, avg_t3))


class SLAMcoreScene(torch.utils.data.Dataset):
    def __init__(self, scene, skip=10, load_all=True, load_last=False, H=480, W=640, center_crop=True, seg_classes="scannet20"):
        # 3D mesh verts
        scene_data_root = get_slamcore_root()
        self.scene_dir = os.path.join(scene_data_root, scene)
        self.mesh_file = os.path.join(
            self.scene_dir, "global_map_mesh.clean.ply")
        ply_dict = get_mesh_vt(self.mesh_file)
        verts_np, faces_np = ply_dict["verts"], ply_dict["faces"]
        self.verts = torch.from_numpy(verts_np)
        self.faces = torch.from_numpy(faces_np)

        self.skip = skip // 10
        self.H, self.W = H, W
        self.H_orig, self.W_orig = 480, 848
        self.center_crop = center_crop
        self.seg_classes = seg_classes
        if self.seg_classes.lower() == "nyu40":
            self.num_classes = 41
        elif self.seg_classes.lower() == "scannet20":
            self.num_classes = 21
        elif self.seg_classes.lower() == "scannet_mixed":
            self.num_classes = 22  # 21 + person
        else:
            raise NotImplementedError
        self.color_encoding = self.get_color_encoding()
        self.transforms = get_transforms(phase="test")
        self.load_all = load_all
        self.load_last = load_last
        self.scene = scene

        # get paths to all the images, depths and labels
        self.rgb_paths = []
        self.depth_paths = []
        self.label_paths = []
        self.frame_ids = []
        self.c2w_list = []
        self.w2c_list = []

        self.slamcore_root = get_slamcore_root()
        self.data_dir = os.path.join(self.slamcore_root, self.scene)
        rgb_dir = os.path.join(self.data_dir, "color")
        depth_dir = os.path.join(self.data_dir, "depth")
        pose_dir = os.path.join(self.data_dir, "pose")
        frames = sorted([int(fid[:-4]) for fid in os.listdir(rgb_dir)])
        self.frames = frames[::self.skip]
        if frames[-1] not in self.frames:
            self.frames.append(frames[-1])

        self.K = np.loadtxt(os.path.join(self.data_dir, "K.txt"))
        if self.H != self.H_orig or self.W != self.W_orig:
            self.scale_x, self.scale_y = self.W / self.W_orig, self.H / self.H_orig
            self.K[0, :] *= self.scale_x
            self.K[1, :] *= self.scale_y
        else:
            self.scale_x, self.scale_y = 1, 1

        if self.center_crop:  # crop horizontally, such that resolution becomes 4:3
            self.W_crop = int(self.H * 4 / 3)
            self.offset_x = (self.W - self.W_crop) // 2
            self.K[0, 2] -= self.offset_x
        else:
            self.W_crop = self.W
            self.offset_x = 0

        for i in self.frames:
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
            self.w2c_list.append(np.linalg.inv(c2w))

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
        rgb = np.array(imageio.imread(rgb_path)).astype(
            np.float32)  # [H, W, 3]
        if self.H != self.H_orig or self.W != self.W_orig:
            rgb = cv2.resize(rgb, (self.W, self.H),
                             interpolation=cv2.INTER_AREA)
        if self.center_crop:
            rgb = rgb[:, self.offset_x:self.offset_x + self.W_crop, :]

        # load depth
        depth = np.array(imageio.imread(depth_path)
                         ).astype(np.float32) / 1000.0
        if self.H != self.H_orig or self.W != self.W_orig:
            depth = cv2.resize(depth, (self.W, self.H),
                               interpolation=cv2.INTER_NEAREST)  # [H, W]
        if self.center_crop:
            depth = depth[:, self.offset_x:self.offset_x + self.W_crop]

        sample = {
            "rgb": rgb,
            "depth": depth,
            "frame_id": self.frame_ids[idx]
        }

        if self.load_all:
            sample["K"] = self.K
            sample["c2w"] = self.c2w_list[idx]
            sample["w2c"] = self.w2c_list[idx]

        sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.rgb_paths)


@torch.no_grad()
def color_mesh(exp_dir, scene, max_frame=0, seg_classes="scannet20", window_size=3, skip=10, eps=0.20, epoch=19, H=480, W=848, center_crop=True, fusion_mode="bayesian"):
    slamcore_root = get_slamcore_root()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    slamcore_scene = SLAMcoreMultiViewDataset(os.path.join(slamcore_root, scene),
                                              max_frame=max_frame,
                                              seg_classes=seg_classes,
                                              window_size=window_size,
                                              skip=skip, H=H, W=W,
                                              center_crop=center_crop,
                                              load_all=True)
    W = slamcore_scene.W_crop
    label_save_dir = os.path.join(
        exp_dir, "slamcore/sequential_new/{}/{:04d}".format(scene, max_frame))
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    print("Processing {}...".format(scene))
    mesh_file = os.path.join(
        label_save_dir, "mesh_{:04d}.ply".format(max_frame))
    ply_dict = get_mesh_vt(mesh_file)
    verts_np, faces_np = ply_dict["verts"], ply_dict["faces"]
    verts = torch.from_numpy(verts_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)

    # For now only use vertex-wise: simple projection
    V, _ = verts.shape
    vert_color = torch.zeros(V, 3, device=device)
    weights = torch.zeros(V, device=device)
    for i in tqdm(range(len(slamcore_scene))):
        frame = slamcore_scene[i]
        c2w, w2c, K, rgb, depth = \
            frame["c2w"].to(device), \
            frame["w2c"].to(device), \
            frame["K"].to(device), \
            frame["rgb"].to(device), \
            frame["depth"].to(device)
        # print("Processing frame: {}".format(frame_id))
        if torch.isnan(c2w).any().item() or torch.isinf(c2w).any().item():
            # print("Skipping frame: {}".format(frame_id))
            continue

        uv_norm, valid_mask = project_pcd(verts.squeeze(), K, H, W, depth=depth.squeeze()[
                                          0], crop=0, w2c=w2c[0], eps=eps)
        valid_uv_norm = uv_norm[valid_mask, :]

        rgb = rgb.squeeze(0)[0]  # [21, H_test, W_test]
        rgb = (rgb * PIXEL_STD.view(3, 1, 1).to(rgb) +
               PIXEL_MEAN.view(3, 1, 1).to(rgb))
        likelihood = F.grid_sample(rgb.unsqueeze(0), valid_uv_norm.unsqueeze(0).unsqueeze(
            0), align_corners=False, padding_mode="border").squeeze().t()  # [N_valid, 21]
        vert_color[valid_mask, :] = (weights[valid_mask].unsqueeze(
            -1) * vert_color[valid_mask, :] + likelihood) / (weights[valid_mask].unsqueeze(-1) + 1.)
        weights[valid_mask] += 1.

    torch.save(vert_color, os.path.join(label_save_dir, "vert_color.pth"))
    tex = TexturesVertex(verts_features=vert_color[None])
    # Only accepts batched input: [B, V, 3], [B, F, 3]
    mesh = Meshes(verts=verts[None], faces=faces[None], textures=tex)
    # For some reason, colors_as_uint8=True is required to save texture
    IO().save_mesh(mesh, os.path.join(label_save_dir,
                                      "{:04d}_vert_color.ply".format(max_frame)), colors_as_uint8=True)


def create_o3d_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                     o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    return mesh


def compute_normals_o3d(verts, faces):
    return np.asarray(create_o3d_mesh(verts, faces).vertex_normals)


def create_3d_data_for_sequential_experiment(exp_dir, scene, skip=10, th_bad_seg=10, k=64):
    slamcore_root = get_slamcore_root()

    ply_dict = get_mesh_vt(os.path.join(slamcore_root, scene, "global_map_mesh.clean.ply"))
    verts, faces, colors = ply_dict["verts"], ply_dict["faces"], ply_dict["colors"]
    if os.path.exists(os.path.join(slamcore_root, scene, "mesh_normals.pth")):
        normals = torch.load(os.path.join(
            slamcore_root, scene, "mesh_normals.pth")).cpu().numpy()
    else:
        normals = compute_normals_o3d(verts, faces)
        torch.save(torch.from_numpy(normals), os.path.join(
            slamcore_root, scene, "mesh_normals.pth"))

    scene_dir = os.path.join(exp_dir, "slamcore/{}_skip{}".format(scene, skip))
    frames = sorted(os.listdir(scene_dir))
    offset_xyz = np.loadtxt(os.path.join(
        slamcore_root, scene, "align.txt"))[None, :]
    T1, T2 = [], []
    for frame in tqdm(frames):
        frame_dir = os.path.join(scene_dir, frame)
        segments = torch.load(os.path.join(
            frame_dir, "segments.pth")).cpu().numpy()
        class_prob = torch.load(os.path.join(
            frame_dir, "class_prob_bayesian.pth")).cpu().numpy()
        n_classes = class_prob.shape[1]
        # count -1 as invalid segment as well
        seg_ids_all = np.unique(segments)
        N_seg_all = len(seg_ids_all)
        valid_segments_mask = np.zeros(
            (N_seg_all,)).astype(bool)  # [N_seg_all,]
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
            seg_center[i] = np.mean(verts[seg_mask, :], axis=0) - offset_xyz
            # cov-matrix
            seg_cov[i] = np.cov(verts[seg_mask, :].transpose())
            # normal
            mean_normals = np.mean(normals[seg_mask, :], axis=0)  # [3,]
            mean_normals = mean_normals / np.linalg.norm(mean_normals)
            # center, color, prob
            mean_colors = np.mean(colors[seg_mask, :], axis=0)  # [3,]
            mean_centers = np.mean(
                verts[seg_mask, :], axis=0) - offset_xyz  # [3,]
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
        T1.append(t2 - t1)
        # save valid_segments_mask and KNN matrix
        torch.save(torch.from_numpy(valid_segments_mask), os.path.join(
            frame_dir, "valid_segments_mask.pth"))  # [N_seg_all]
        # create KD-tree
        t3 = get_time()
        tree = KDTree(seg_center)
        # Store NN-matrix instead of hard-coded K-NN features
        nn_mat = tree.query(seg_center, k=min(k, len(seg_center)), return_distance=False, sort_results=True)  # [N_seg, N_seg]
        t4 = get_time()
        T2.append(t4 - t3)
        torch.save(torch.from_numpy(nn_mat), os.path.join(
            frame_dir, "nn_mat.pth"))  # [N_seg, N_seg]
        # save segment features
        torch.save(torch.from_numpy(seg_center), os.path.join(
            frame_dir, "seg_center.pth"))  # [N_seg, 3]
        torch.save(torch.from_numpy(seg_cov), os.path.join(
            frame_dir, "seg_cov.pth"))  # [N_seg, 3, 3]
        torch.save(torch.from_numpy(seg_feat_prob), os.path.join(
            frame_dir, "seg_feat_prob.pth"))  # [N_seg, C]


class SLAMcoreSceneSegments(torch.utils.data.Dataset):
    def __init__(self, exp_dir, scene, skip=20, k=64):
        self.scene_dir = os.path.join(exp_dir, "slamcore/{}_skip{}".format(scene, skip))
        frame_list = sorted(os.listdir(self.scene_dir))
        # Need to construct K-NN on-the-fly
        self.seg_center_list = []
        self.seg_cov_list = []
        self.seg_feat_list = []
        self.knn_mat_list = []
        self.frame_id_list = []

        for frame in frame_list:
            frame_dir = os.path.join(self.scene_dir, frame)
            self.seg_center_list.append(torch.load(os.path.join(frame_dir, "seg_center.pth")).float())  # [N_seg, 3]
            self.seg_cov_list.append(torch.load(os.path.join(frame_dir, "seg_cov.pth")).float())  # [N_seg, 3, 3]
            # input feature
            feats = torch.load(os.path.join(
                frame_dir, "seg_feat_prob.pth")).float()  # [N_seg, C]
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


def apply_3dconv_for_sequental_experiment(exp_dir, scene, skip=20, k=64, epoch="best", device=torch.device("cuda:0")):
    log_dir = os.path.join(exp_dir, "SegConvNet")
    cfg_file = os.path.join(log_dir, "config.yaml")
    cfg = load_yaml(cfg_file)
    model = get_segconv_model(cfg)
    model.to(device)
    chkpt = torch.load(os.path.join(log_dir, "checkpoints/chkpt-{}.pth".format(epoch)), map_location=device)
    model.load_state_dict(chkpt["state_dict"])
    model.eval()
    mesh_file = os.path.join(get_slamcore_root(), scene, "global_map_mesh.clean.ply")
    scene_data = SLAMcoreSceneSegments(exp_dir, scene, skip=skip, k=k)

    T = []
    for frame_data in tqdm(scene_data):
        xyz = frame_data["locs"].unsqueeze(0).to(device)  # [1, N_seg, 3]
        cov = frame_data["covs"].unsqueeze(0).to(device)  # [1, N_seg, 3, 3]
        feat = frame_data["feats"].unsqueeze(0).to(device)  # [1, N_seg, C]
        knn_indices = frame_data["knn_indices"].to(device)  # [N_seg, K]
        frame_id = frame_data["frame_id"]

        # Forward propagation
        t1 = get_time()
        # [n_classes, N_seg]
        out = model(xyz, cov, feat, knn_indices).squeeze()
        seg_label_pred = torch.argmax(out, dim=0).cpu().numpy()  # [N_seg,]
        t2 = get_time()
        T.append(t2 - t1)

        # load segment.pth
        frame_dir = os.path.join(
            exp_dir, "slamcore/{}_skip{}/{:06d}".format(scene, skip, frame_id))
        segments = torch.load(os.path.join(frame_dir, "segments.pth")).cpu().numpy()
        verts_label_pred = torch.load(os.path.join(frame_dir, "class_label_bayesian.pth")).cpu().numpy()  # [V,]
        assert len(segments) == len(
            verts_label_pred), "Number of vertices mismatch!!!"
        seg_ids_all = np.unique(segments)  # [N_seg_all,]
        if os.path.exists(os.path.join(frame_dir, "valid_segments_mask.pth")):
            valid_segments_mask = torch.load(os.path.join(
                frame_dir, "valid_segments_mask.pth")).cpu().numpy()  # [N_seg_all,]
            seg_ids = seg_ids_all[valid_segments_mask]  # [N_seg]
        else:
            seg_ids = seg_ids_all
        assert len(seg_label_pred) == len(
            seg_ids), "Number of segments mismatch!!!"
        N_seg = len(seg_label_pred)

        # 0 to N_seg - 1
        for i, seg_id in enumerate(seg_ids):
            # in very rare cases, the seg_ids is not range(N_seg), but still don't know why...
            seg_mask = (segments == seg_id)
            verts_label_pred[seg_mask] = seg_label_pred[i]
        save_dir = os.path.join(frame_dir, "final_results")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(torch.from_numpy(verts_label_pred),
                   os.path.join(save_dir, "verts_label.pth"))
        # save labelled mesh
        ply_dict = get_mesh_vt(mesh_file)
        verts, faces = ply_dict["verts"], ply_dict["faces"]
        vert_colors = vert_label_to_color(verts_label_pred, color_encoding_scannetX)
        assert verts.shape[0] == verts_label_pred.shape[0], "Number of vertices doesn't match!!!"
        mesh_save = trimesh.Trimesh(verts, faces, vertex_colors=vert_colors)
        mesh_save.export(os.path.join(save_dir, "verts_label.ply"))
        obs_mask = torch.load(os.path.join(
            frame_dir, "obs_mask.pth")).cpu().numpy()

        valid_faces_mask = (
            obs_mask[faces[:, 0]] | obs_mask[faces[:, 1]] | obs_mask[faces[:, 2]])
        valid_faces = faces[valid_faces_mask, :]
        cleaned_mesh = trimesh.Trimesh(
            verts, valid_faces, vertex_colors=vert_colors)
        cleaned_mesh.remove_unreferenced_vertices()
        cleaned_mesh.export(os.path.join(save_dir, "verts_label_clean.ply"))


def run_sequential_mapping():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--scene", type=str, default="kitchen1")
    parser.add_argument("--model_type", type=str, default="LPN_rgb")
    parser.add_argument("--mapping_every", type=int, default=20)
    args = parser.parse_args()
    exp_dir = args.exp_dir
    scene = args.scene
    model_type = args.model_type

    # run_sequential_qpos(exp_dir, scene, mapping_every=args.mapping_every)
    bayesian_runner = BayesianLabelSequential(exp_dir, scene,
                                              model_type=model_type,
                                              center_crop=True,
                                              skip=args.mapping_every)
    bayesian_runner.run(eps=0.20)
    create_3d_data_for_sequential_experiment(exp_dir, scene, skip=args.mapping_every)
    apply_3dconv_for_sequental_experiment(exp_dir, scene)


if __name__ == "__main__":
    run_sequential_mapping()
