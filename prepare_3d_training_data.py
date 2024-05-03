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
import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import KDTree
from tqdm import tqdm
import ray
import trimesh

from qpos.segment_mesh import label_segments
from dataio.utils import get_scene_list, vert_label_to_color, color_encoding_scannet20, read_ply


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def create_o3d_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                     o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    return mesh


def compute_normals_o3d(verts, faces):
    return np.asarray(create_o3d_mesh(verts, faces).vertex_normals)


def save_as_mesh(verts, faces, labels, save_path):
    assert verts.shape[0] == labels.shape[0], "Number of vertices doesn't match!!!"
    vert_colors = vert_label_to_color(labels, color_encoding_scannet20)
    mesh_save = trimesh.Trimesh(verts, faces, vertex_colors=vert_colors)
    mesh_save.export(os.path.join(save_path))


def colorize_segments(verts, faces, segments, segment_colors):
    """
    :param verts: [V, 3]
    :param faces: [F, 3]
    :param segments: [V,]
    :param segment_colors: [N_seg, 3]
    :return:
    """

    vert_colors = np.zeros_like(verts)
    seg_ids = np.unique(segments)
    for i, seg_id in enumerate(seg_ids):
        vert_colors[segments == seg_id, :] = segment_colors[i, :]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vert_colors)
    return mesh


def process_one_scene(scene,
                      dataset_type,
                      dataset_root,
                      label_fusion_dir,  # dir of saving bayesian-fused meshes
                      segment_suffix,
                      mesh_name,
                      save_mesh,
                      load_label=True,
                      o3d_normal=True):
    # Input:
    # mesh_verts: [V, 3] vertex positions
    # segments: [V,] stores segment id for each vertex
    # label_major_vote: [V,] stores gt class id (obtained via major vote at segment_v2 level) for each vertex
    # class_label_bayesian: [V,] fused predicted class id
    # class_prob_bayesian: [V, 21] fused predicted class logits
    # Output:
    # segment centroids: [N_seg, 3]
    # segment features: [N_seg, C=n_classes+6]
    # segment labels: [N_seg,]
    # Also need K-NN

    scene_dir = os.path.join(dataset_root, scene)
    scene_segment_dir = os.path.join(dataset_root, scene, segment_suffix)
    scene_label_fusion_dir = os.path.join(label_fusion_dir, scene)
    save_dir = os.path.join(scene_label_fusion_dir, segment_suffix)
    os.makedirs(save_dir, exist_ok=True)
    if dataset_type == "scannet_test":
        load_label = False

    mesh_path = os.path.join(scene_dir, mesh_name)
    mesh_data = read_ply(mesh_path)
    verts = mesh_data["verts"]
    vert_colors = mesh_data["colors"]
    faces = mesh_data["faces"]
    vert_normals = mesh_data["normals"]
    if vert_normals is None or o3d_normal:
        vert_normals = compute_normals_o3d(verts, faces)

    segments = torch.load(os.path.join(scene_segment_dir, "segments.pth")).cpu().numpy()  # [V,]
    # mesh label predicted with LPN + Bayesian-Labelling
    pred_label_file = "class_label_bayesian.pth"
    label_pred = torch.load(os.path.join(scene_label_fusion_dir, pred_label_file), map_location=torch.device("cpu")).numpy()
    # mesh class probability with LPN + Bayesian-Labelling
    pred_prob_file = "class_prob_bayesian.pth"
    prob_pred = torch.load(os.path.join(scene_label_fusion_dir, pred_prob_file), map_location=torch.device("cpu")).numpy()

    # gt segment label by major vote
    if load_label:
        label_major_vote_gt = torch.load(os.path.join(scene_segment_dir, "label_major_vote.pth")).cpu().numpy()
        assert verts.shape[0] == vert_colors.shape[0] == vert_normals.shape[0] == segments.shape[0] == label_major_vote_gt.shape[0] == label_pred.shape[0] == prob_pred.shape[0], "Shape mismatch!"
    else:
        label_major_vote_gt = None
        assert verts.shape[0] == vert_colors.shape[0] == vert_normals.shape[0] == segments.shape[0] == label_pred.shape[0] == prob_pred.shape[0], "Shape mismatch!"

    n_classes = prob_pred.shape[-1]
    # raw segments that might contain outlier segments
    seg_ids_all = np.unique(segments)  # [N_seg_all,]
    N_seg_all = seg_ids_all.shape[0]
    valid_segments_mask = np.zeros((N_seg_all,)).astype(bool)  # [N_seg_all,]
    seg_center = np.zeros((N_seg_all, 3))  # [N_seg_all, 3]
    seg_cov = np.zeros((N_seg_all, 3, 3))  # [N_seg_all, 3, 3]
    seg_label = np.zeros((N_seg_all,))  # [N_seg_all,]
    seg_feat_prob = np.zeros((N_seg_all, n_classes + 9))  # [N_seg_all, C]
    seg_feat_label = np.zeros((N_seg_all, n_classes + 9))  # [N_seg_all, C]

    # get major-vote result for predicted labels
    # save majority-vote meshes, not necessary for training
    if save_mesh:
        label_major_vote_pred = label_segments(segments, label_pred, respect_unknown=False)
        save_as_mesh(verts, faces, label_major_vote_pred, os.path.join(save_dir, "label_major_vote_pred.ply"))
        torch.save(torch.from_numpy(label_major_vote_pred), os.path.join(save_dir, "label_major_vote_pred.pth"))
        if label_major_vote_gt is not None:
            save_as_mesh(verts, faces, label_major_vote_gt, os.path.join(save_dir, "label_major_vote_gt.ply"))
            torch.save(torch.from_numpy(label_major_vote_gt), os.path.join(save_dir, "label_major_vote_gt.pth"))

    for i, seg_id in enumerate(seg_ids_all):
        # print(i == seg_id)
        seg_mask = (segments == seg_id)

        # Skip bad segments
        if len(verts[seg_mask, :]) < 10:
            # print("TOO SMALL SEGMENT id {} with {} vertices".format(i, len(verts[seg_mask, :])))
            valid_segments_mask[i] = False
            continue

        valid_segments_mask[i] = True
        # center
        seg_center[i] = np.mean(verts[seg_mask, :], axis=0)
        # cov-matrix
        seg_cov[i] = np.cov(verts[seg_mask, :].transpose())
        # normal
        mean_normals = np.mean(vert_normals[seg_mask, :], axis=0)  # [3,]
        mean_normals = mean_normals / np.linalg.norm(mean_normals)
        # center, color, prob
        mean_colors = np.mean(vert_colors[seg_mask, :], axis=0)  # [3,]
        mean_centers = np.mean(verts[seg_mask, :], axis=0)  # [3,]
        mean_probs = np.mean(prob_pred[seg_mask, :], axis=0)  # [21,]
        # feature_prob
        seg_feat_prob[i, :3] = mean_colors
        seg_feat_prob[i, 3:6] = mean_normals
        seg_feat_prob[i, 6:9] = mean_centers
        seg_feat_prob[i, 9:] = mean_probs

        seg_labels_pred = label_pred[seg_mask]  # [seg_size]
        seg_size = seg_labels_pred.shape[0]
        mean_labels = np.zeros((n_classes,))
        for class_id in range(n_classes):
            mean_labels[class_id] = np.count_nonzero(seg_labels_pred == class_id) / seg_size
        seg_feat_label[i, :3] = mean_colors
        seg_feat_label[i, 3:6] = mean_normals
        seg_feat_label[i, 6:9] = mean_centers
        seg_feat_label[i, 9:] = mean_labels

        # gt label
        if load_label:
            label_gt = label_major_vote_gt[seg_mask]  # should be consistent
            assert np.unique(label_gt).shape[0] == 1, "Encountered inconsistent labels for segment!"
            seg_label[i] = label_gt[0]

    # save colored segments (for sanity check)
    mesh_debug = colorize_segments(verts, faces, segments, seg_feat_label[:, :3])
    mesh_debug.export(os.path.join(save_dir, "segments_color.ply"))

    if np.count_nonzero(~valid_segments_mask) > 0:  # If there are invalid segment
        # save bad segments (if any) for sanity check
        outlier_colors = np.zeros((N_seg_all, 3))
        outlier_colors[valid_segments_mask, :] = np.array([0., 0., 0.])  # black for inlier
        outlier_colors[~valid_segments_mask, :] = np.array([1., 0., 0.])  # red for outlier
        mesh_outlier = colorize_segments(verts, faces, segments, outlier_colors)
        mesh_outlier.export(os.path.join(save_dir, "segments_outlier.ply"))

        # get valid segments only
        N_seg = np.count_nonzero(valid_segments_mask)
        seg_center = seg_center[valid_segments_mask, :]
        seg_cov = seg_cov[valid_segments_mask, :, :]
        seg_label = seg_label[valid_segments_mask]
        seg_feat_prob = seg_feat_prob[valid_segments_mask, :]
        seg_feat_label = seg_feat_label[valid_segments_mask, :]
    else:
        N_seg = N_seg_all

    # save valid_segments_mask
    torch.save(torch.from_numpy(valid_segments_mask), os.path.join(save_dir, "valid_segments_mask.pth"))  # [N_seg_all]
    # create KD-tree
    tree = KDTree(seg_center)
    # Store NN-matrix
    nn_mat = tree.query(seg_center, k=N_seg, return_distance=False, sort_results=True)  # [N_seg, N_seg]
    torch.save(torch.from_numpy(nn_mat), os.path.join(save_dir, "nn_mat.pth"))  # [N_seg, N_seg]

    # per-segment data
    torch.save(torch.from_numpy(seg_center), os.path.join(save_dir, "seg_center.pth"))  # [N_seg, 3]
    torch.save(torch.from_numpy(seg_cov), os.path.join(save_dir, "seg_cov.pth"))  # [N_seg, 3, 3]
    torch.save(torch.from_numpy(seg_feat_prob), os.path.join(save_dir, "seg_feat_prob.pth"))  # [N_seg, C]
    torch.save(torch.from_numpy(seg_feat_label), os.path.join(save_dir, "seg_feat_label.pth"))  # [N_seg, C]

    if load_label:
        torch.save(torch.from_numpy(seg_label), os.path.join(save_dir, "seg_label.pth"))  # gt segmant label [N_seg]


def process_scenes(scene_list, dataset_type, dataset_root, label_fusion_dir, segment_suffix, mesh_name_suffix, save_mesh):
    for scene in tqdm(scene_list):
        mesh_name = scene + mesh_name_suffix
        process_one_scene(scene, dataset_type, dataset_root, label_fusion_dir, segment_suffix, mesh_name, save_mesh)


def process_all_scenes():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_fusion_dir", type=str, required=True)
    parser.add_argument("--segment_suffix", type=str, required=True, help="sub_dir to save the segments")
    parser.add_argument("--dataset_type", type=str, help="dataset type, scannet, scannet_test or slamcore", default="scannet")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--save_mesh", dest="save_mesh", action="store_true", help="Save mesh for visualisation")
    parser.add_argument("--n_proc", type=int, default=12)
    args = parser.parse_args()

    # get args
    dataset_type = args.dataset_type
    dataset_root = args.dataset_root
    label_fusion_dir = args.label_fusion_dir
    segment_suffix = args.segment_suffix
    save_mesh = args.save_mesh

    n_proc = args.n_proc
    process_scenes_remote = ray.remote(process_scenes)
    ray.init()

    print(dataset_type)
    if dataset_type == "scannet":
        scene_split_file = "configs/scannetv2_trainval.txt"
        scene_list = get_scene_list(scene_split_file)
        scene_lists = split_list(scene_list, n_proc)
        mesh_name_suffix = "_vh_clean_2.labels.ply"
    elif dataset_type == "scannet_test":
        scene_split = "configs/scannetv2_test.txt"
        scene_list = get_scene_list(scene_split)
        scene_lists = split_list(scene_list, n_proc)
        mesh_name_suffix = "_vh_clean_2.ply"
    elif dataset_type == "slamcore":
        scene_list = get_scene_list("configs/slamcore.txt")
        n_proc = 4
        scene_lists = split_list(scene_list, n_proc)
        mesh_name_suffix = "_clean.labels.ply"

    futures = [process_scenes_remote.remote(scene_lists[w_id], dataset_type, dataset_root, label_fusion_dir, segment_suffix, mesh_name_suffix, save_mesh) for w_id in range(n_proc)]
    ray.get(futures)


if __name__ == "__main__":
    process_all_scenes()
