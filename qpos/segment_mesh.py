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
import numpy as np
from sklearn.neighbors import KDTree
import time
import trimesh
import torch
import open3d as o3d

from qpos.geometric_instances import (
    segment_mesh,
    get_vertex_root,
    normalize_v3,
    merge_small_segments
)
from dataio.utils import nyu40_to_scannet20, vert_label_to_color, color_encoding_scannet20, read_ply


# implementation adopted from CUDA code in solid_mesh_color.vert,
# taken from https://www.shadertoy.com/view/llGSzw
def hash_colour(n):
    n = np.int64(n)
    n = (n << 13) ^ n
    n = n * (n * n * 15731 + 789221) + 1376312589
    k = np.zeros(3, dtype=np.int64)
    k[0] = n * n
    k[1] = n * n * 16807
    k[2] = n * n * 48271
    r = np.zeros(3, dtype=np.int64)
    for i in range(0, 3):
        r[i] = 255 * ((k[i] & 0x7FFFFFFF) / float(0x7FFFFFFF))
    return r


def colour_map_using_hash(max_seg):
    colour_map = np.zeros((max_seg + 1, 3))
    for i in range(0, max_seg + 1):
        colour_map[i] = hash_colour(i)
    return colour_map


def colourize_instances(plydata, colour_map, instances, mask=None, min_value=1):
    """
    Colourize instances in a point cloud
    :param plydata:
    :param colour_map:
    :param instances:
    :param mask:
    :return:
    """
    if mask is None:
        mask = np.ones(len(instances), dtype=np.bool)
    for el in plydata.elements:
        if el.name == "vertex":
            inst_inds = (instances >= min_value) & mask
            el.data["red"] = 255
            el.data["green"] = 255
            el.data["blue"] = 255

            el.data["red"][inst_inds] = colour_map[instances[inst_inds], 0]
            el.data["green"][inst_inds] = colour_map[instances[inst_inds], 1]
            el.data["blue"][inst_inds] = colour_map[instances[inst_inds], 2]


def compute_normals(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)
    return norm


def create_o3d_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                     o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    return mesh


def compute_normals_o3d(verts, faces):
    return np.asarray(create_o3d_mesh(verts, faces).vertex_normals)


def estimate_normals_from_depth(depth, points_world, points_mask):
    h, w = depth.shape[0:2]
    points_world_map = points_world.reshape((h, w, 3))
    points_world_mask = points_mask.reshape((h, w))
    points_right_map = points_world_map[:, 1:]
    points_right_mask = points_world_mask[:, 1:]
    points_bottom_map = points_world_map[1:, :]
    points_bottom_mask = points_world_mask[1:, :]
    dpdx = points_right_map[:-1, :] - points_world_map[:-1, :-1]
    dpdx_mask = points_right_mask[:-1, :] & points_world_mask[:-1, :-1]
    dpdy = points_bottom_map[:, :-1] - points_world_map[:-1, :-1]
    dpdy_mask = points_bottom_mask[:, :-1] & points_world_mask[:-1, :-1]
    normals_map = np.zeros((h, w, 3))
    normals = -np.cross(dpdx.reshape(-1, 3), dpdy.reshape(-1, 3))
    normalize_v3(normals)
    normals_map[:-1, :-1, :] = normals.reshape((h-1, w-1, 3))
    normals_mask = np.zeros((h, w), dtype=np.bool)
    normals_mask[:-1, :-1] = dpdx_mask & dpdy_mask
    return normals_map.reshape(-1, 3), normals_mask.reshape(-1)


def oversegmentation_statistics(verts, normals, seg_labels, class_labels):
    """
    Print the statistics of the segments, show histograms
    :param verts: vertices [V, 3]
    :param normals: normals [V, 3]
    :param seg_labels: segment_v1 id per vertex [V,]
    :param class_labels: gt class id per vertex [V,]
    """
    print("Start stats...")
    t_start = time.time()
    seg_ids = np.unique(seg_labels)  # [N_seg,]
    n_seg = len(seg_ids)
    print("In total, {} segments.".format(n_seg))
    d_spat = np.zeros(n_seg)  # diameter of segments
    d_ang = np.zeros(n_seg)  # angle of segments
    trcov_spat = np.zeros(n_seg)
    trcov_ang = np.zeros(n_seg)
    # min_normals = get_minimal_normal(normals)
    seg_sizes = np.zeros(n_seg)  # segment_v1 sizes (num of vertices)
    seg_id_size = np.zeros(np.max(seg_ids) + 1)  # same but organised as seg_ids
    for i, sid in enumerate(seg_ids):
        mask = seg_labels == sid  # [V,]
        # if np.sum(mask) <= 1:
        #     continue

        # compute spatial dimension (diameter) of the segment_v1
        d_spat[i] = compute_diameter(verts[mask])
        spat_cov = np.cov(verts[mask].T)  # 3x3 spatial covariance matrix of the segment_v1
        trcov_spat[i] = np.sqrt(spat_cov.trace())  # sqrt of the trace of the covariance matrix
        d_norm = compute_diameter(normals[mask])  # max chord length of the direction difference
        d_ang[i] = 2 * np.arcsin(np.min([d_norm / 2, 1.0])) * 180.0 / np.pi  # max angle-difference in normals
        norm_cov = np.cov(normals[mask].T)
        trcov_ang[i] = np.sqrt(norm_cov.trace())
        seg_sizes[i] = np.sum(mask)
        seg_id_size[sid] = seg_sizes[i]

    large_seg_size_thr = 10
    large_seg_mask = seg_sizes > large_seg_size_thr
    large_seg_id_mask = seg_id_size > large_seg_size_thr
    large_seg_vert_mask = large_seg_id_mask[seg_labels]
    print("Large segments: {}".format(np.sum(large_seg_mask)))
    print("Stats in {} sec.".format(time.time() - t_start))
    # plt.figure(0)
    # plt.title('Spatial diameter')
    # plt.hist(d_spat[large_seg_mask])
    print(
        "Spatial diameter: avg {} sigma {} max {}".format(
            np.mean(d_spat), np.std(d_spat), np.max(d_spat)
        )
    )
    print(
        "Spatial diameter (large): avg {} sigma {} max {}".format(
            np.mean(d_spat[large_seg_mask]),
            np.std(d_spat[large_seg_mask]),
            np.max(d_spat[large_seg_mask]),
        )
    )
    print(
        "Spatial cov trace sqrt: avg {} max {}".format(
            np.mean(trcov_spat), np.max(trcov_spat)
        )
    )
    print(
        "Size: avg {} sigma {} max {}".format(
            np.mean(seg_sizes),
            np.std(seg_sizes),
            np.max(seg_sizes)
        )
    )
    # plt.figure(1)
    # plt.title('Normals angular diameter')
    # plt.hist(d_ang[large_seg_mask])
    print(
        "Normals angular diameter (deg.): avg {} sigma {} max {}".format(
            np.mean(d_ang), np.std(d_ang), np.max(d_ang)
        )
    )
    print(
        "Normals angular diameter (large, deg.): avg {} sigma {} max {}".format(
            np.mean(d_ang[large_seg_mask]),
            np.std(d_ang[large_seg_mask]),
            np.max(d_ang[large_seg_mask]),
        )
    )
    print(
        "Normals cov trace sqrt: avg {} max {}".format(
            np.mean(trcov_ang), np.max(trcov_ang)
        )
    )
    seg_sizes_srt = np.asarray(sorted(seg_sizes, reverse=True))  # descending order
    size_cdf = np.cumsum(seg_sizes_srt)
    size_cdf = size_cdf / np.sum(seg_sizes)
    # plt.figure(2)
    # plt.title('Segment size CDF')
    # plt.plot(-seg_sizes_srt, size_cdf)
    ones_arr = np.ones_like(seg_sizes)
    ind_100 = np.argmin(np.abs(seg_sizes_srt - 100 * ones_arr))
    ind_50 = np.argmin(np.abs(seg_sizes_srt - 50 * ones_arr))
    ind_10 = np.argmin(np.abs(seg_sizes_srt - 10 * ones_arr))
    print(
        "Segs > 100 vertices: {:.2f}%, >50 vertices: {:.2f}%, >10 vertices: {:.2f}% ".format(
            size_cdf[ind_100] * 100, size_cdf[ind_50] * 100, size_cdf[ind_10] * 100
        )
    )

    if class_labels is not None:
        # Should respect_unknown be True or False? I guess False makes more sense?
        labels_majvote = label_segments(seg_labels, class_labels, respect_unknown=False)
        labels_err = np.abs(labels_majvote - class_labels)
        lbl_err_mask = labels_err > 0
        lbl_err_mask[class_labels == unknown_label] = 0
        oversegmentation_labelling_error = np.sum(lbl_err_mask) / len(lbl_err_mask) * 100
        print(
            "Error in labels: {:.2f}%".format(
                oversegmentation_labelling_error
            )
        )
        print(
            "Error in labels for large: {:.2f}%".format(
                np.sum(lbl_err_mask[large_seg_vert_mask])
                / len(lbl_err_mask[large_seg_vert_mask])
                * 100
            )
        )
    else:
        labels_majvote, lbl_err_mask, oversegmentation_labelling_error = None, None, None

    small_mask_1 = seg_sizes <= 1
    small_mask_3 = seg_sizes <= 3
    small_mask_5 = seg_sizes <= 5
    small_mask_10 = seg_sizes <= 10

    return (
        labels_majvote, lbl_err_mask,
        oversegmentation_labelling_error,
        np.mean(seg_sizes),
        np.min(seg_sizes),
        np.max(seg_sizes),
        np.count_nonzero(small_mask_1),
        np.count_nonzero(small_mask_3),
        np.count_nonzero(small_mask_5),
        np.count_nonzero(small_mask_10),
        n_seg,
        np.mean(d_spat)
    )


unknown_label = 0  # 255


def label_segments(segments, labels_init, respect_unknown=False, skip_segment=-1):
    """
    Label the vertices using majority voting over segments

    :param segments: segment_v1 id for each vertex
    :param labels_init: vertex labels
    :return: per-vertex labels obtained from segment_v1-based voting
    """
    seg_ids = np.unique(segments)
    labels = np.copy(labels_init)
    for s_id in seg_ids:
        mask = segments == s_id
        if s_id == skip_segment:
            labels[mask] = 255
            continue
        class_ids, cnt = np.unique(labels[mask], return_counts=True)
        if respect_unknown:
            cnt[class_ids == unknown_label] = 0
        # print('segment_v1 {}: labels '.format(s_id) + str(class_ids))
        max_class_id = np.argmax(cnt)
        labels[mask] = class_ids[max_class_id]
    return labels


def compute_diameter(vects):
    """
    Compute a diameter (maximal distance between vectors) for a vector set
    :param vects: vectors [N, 3]
    :return: diameter
    """
    vvt = vects @ vects.T  # [N, N] x_i^t @ x_j
    vn = np.linalg.norm(vects, axis=1) ** 2
    n = len(vn)
    vnh = np.tile(vn.reshape(-1, 1), (1, n))  # x_j^t @ x_j
    vnv = np.tile(vn.reshape(1, -1), (n, 1))  # x_i^t @ x_j
    dists = vnh + vnv - 2 * vvt  # (x_i - x_j)^t @ (x_i - x_j) = x_i^t @ x_i + x_j^t @ x_j - x_i^t @ x_j
    d_max = np.max(dists.reshape(-1))
    return np.sqrt(np.max([d_max, 0]))


def save_eval_ply(plydata, vert_lbls, out_dir, lbl_err_mask=None, pref=""):
    """
    Save the evaluation as a point cloud

    :param vert_lbls: segment_v1 id per vertex
    :param lbl_err_mask: if a vertex has an error in class label
    :param out_dir: a directory to save
    :param pref: a file prefix for the point cloud
    """
    colour_map_seg = np.random.randint(0, size=(np.max(vert_lbls) + 1, 3), high=255)
    colourize_instances(plydata, colour_map_seg, vert_lbls)
    plydata.write(out_dir + "/" + pref + "_seg.ply")

    if lbl_err_mask is not None:
        colour_map_seg = np.zeros((2, 3))
        colour_map_seg[1] = np.asarray([255, 0, 0])
        colourize_instances(plydata, colour_map_seg, lbl_err_mask.astype(np.uint32))
        plydata.write(out_dir + "/" + pref + "_err.ply")
        plydata.write(out_dir + "/" + pref + "_cov.ply")


def get_segment_statistics(segments, verts, normals):
    # compute segment stats
    segment_ids, segments_new = np.unique(segments, return_inverse=True)
    seg_centers = np.zeros((len(segment_ids), 3))
    seg_normals = np.zeros((len(segment_ids), 3))
    seg_covs = np.zeros((len(segment_ids), 3, 3))
    normal_covs = np.zeros((len(segment_ids), 3, 3))
    seg_sizes = np.zeros((len(segment_ids)))
    for i, segment_id in enumerate(segment_ids):
        segment_mask = (segments == segment_id)
        seg_centers[i] = np.mean(verts[segment_mask, :], axis=0)
        mean_normal = np.mean(normals[segment_mask, :], axis=0)
        seg_normals[i] = mean_normal / (
                np.linalg.norm(mean_normal) + np.finfo(float).eps
        )
        seg_sizes[i] = np.sum(segment_mask)
        if np.sum(segment_mask) < 2:  # no way to estimate covariance...
            continue
        seg_covs[i] = np.cov(verts[segment_mask, :].transpose())
        normal_covs[i] = np.cov(normals[segment_mask, :].transpose())
    # sort by increasing size
    sort_inds = np.argsort(seg_sizes)
    seg_centers = seg_centers[sort_inds]
    seg_normals = seg_normals[sort_inds]
    seg_covs = seg_covs[sort_inds]
    normal_covs = normal_covs[sort_inds]
    seg_sizes = seg_sizes[sort_inds]
    segment_ids = segment_ids[sort_inds]
    inverse_sort_inds = np.zeros((len(sort_inds)), dtype=np.int32)
    inverse_sort_inds[sort_inds] = np.arange(0, len(sort_inds))
    segments_new = inverse_sort_inds[segments_new]
    return (seg_centers,
            seg_normals,
            seg_covs,
            normal_covs,
            seg_sizes,
            segment_ids,
            segments_new
            )


def reproject_segments(verts, normals, seg_centers, seg_covs, seg_normals, k=20, seg_ids=None):
    t_start_reproj = time.time()
    if seg_ids is None:
        seg_ids = np.arange(0, seg_centers.shape[0])    
    tree = KDTree(seg_centers[seg_ids])

    num_neighbours = np.min([k, seg_centers.shape[0]])
    normal_cost_weight = 10.0
    segments_update = np.zeros((verts.shape[0]), dtype=np.int32)
    t_start_query = time.time()
    nn_inds = tree.query(verts, k=num_neighbours, return_distance=False, sort_results=True)
    t_end_query = time.time()
    # here is a batched code for
    # dp = vert - seg_center
    # spatial_cost = dp.T @ inv_seg_cov @ dp
    # normal cost = 1 - <seg_normal, vert_normal>
    # cost = spatial_cost + normal_cost_weight * normal_cost
    num_verts = nn_inds.shape[0]
    curr_seg = seg_ids[nn_inds] # n_verts x num_neighbours
    seg_centers_sampled = seg_centers[curr_seg.reshape(-1)].reshape((num_verts, num_neighbours, 3))
    verts_reshaped = verts.reshape((-1, 1, 3))
    dp_batch = verts_reshaped - seg_centers_sampled
    inv_covs = np.linalg.inv(seg_covs + 1e-10 * np.eye(3).reshape(1, 3, 3))
    inv_covs_batch = inv_covs[nn_inds.reshape(-1)].reshape((num_verts, num_neighbours, 3, 3))
    dp_icprod_batch = np.einsum("ijk,ijkl->ijl", dp_batch, inv_covs_batch)
    costs_batch = np.einsum("ijk,ijk->ij", dp_icprod_batch, dp_batch)

    seg_normals_sampled = seg_normals[nn_inds.reshape(-1)].reshape((num_verts, num_neighbours, 3))
    normal_costs = 1.0 - np.sum(normals.reshape(-1, 1, 3) * seg_normals_sampled, axis=2)
    costs_batch += normal_cost_weight * normal_costs
    min_inds = np.argmin(costs_batch, axis=1)
    best_inds = nn_inds[np.arange(0, num_verts), min_inds]
    segments_update = seg_ids[best_inds]

    t_end_batch = time.time()
    # print(
    # f'Reproj timings: tree {t_start_query - t_start_reproj} query {t_end_query - t_start_query} bloop {t_end_batch - t_end_query} ')
    return segments_update


def split_large_segments(centers_large, normals_large, covs_large, normal_covs_large,
                         seg_sizes_large):
    new_centers = []
    new_normals = []
    new_covs = []
    new_normal_covs = []
    new_seg_sizes = []
    for i in range(0, centers_large.shape[0]):
        cov = covs_large[i]
        values, vectors = np.linalg.eig(cov)
        # print(
        # f'cov err {np.linalg.norm(vectors @ np.diag(values) @ vectors.T - cov)}')
        max_val_ind = np.argmax(values)
        values_reduced = np.copy(values)
        values_reduced[max_val_ind] /= 4.0
        principal_dir = vectors[:, max_val_ind]
        principal_dir /= np.linalg.norm(principal_dir)
        sigma = np.sqrt(values[max_val_ind])
        center = centers_large[i]
        center_one = center + sigma * principal_dir
        center_two = center - sigma * principal_dir
        new_centers.append(center_one)
        new_normals.append(normals_large[i])
        cov_reduced = vectors @ np.diag(values_reduced) @ vectors.T
        new_covs.append(cov_reduced)
        new_normal_covs.append(normal_covs_large[i])
        new_seg_sizes.append(seg_sizes_large[i] / 2.0)
        # second child
        new_centers.append(center_two)
        new_normals.append(normals_large[i])
        new_covs.append(cov_reduced)
        new_normal_covs.append(normal_covs_large[i])
        new_seg_sizes.append(seg_sizes_large[i] / 2.0)

    if len(new_centers) > 0:

        new_centers = np.asarray(new_centers)
        new_normals = np.asarray(new_normals)
        new_covs = np.asarray(new_covs)
        new_normal_covs = np.asarray(new_normal_covs)
        new_seg_sizes = np.asarray(new_seg_sizes)

        return new_centers, new_normals, new_covs, new_normal_covs, new_seg_sizes
    else:
        return centers_large, normals_large, covs_large, normal_covs_large, \
            seg_sizes_large


def quasiplanar_segmentation(verts, normals, faces,
                             is_reprojection=True,
                             expected_segment_size=0.1,
                             small_segment_size=0.02,
                             is_v3=False,
                             **kwargs):
    is_metric = True
    is_merge_small = True
    is_split_large_segments = False
    is_merge_small_reprojection = False
    if is_v3:
        # new improvements ("qpos")
        is_metric = False
        is_merge_small = False
        is_split_large_segments = True
        is_merge_small_reprojection = True

    segments, edges_sorted = segment_mesh(
        faces, normals, verts,  expected_segment_size, small_size=small_segment_size,
        is_merge_small=is_merge_small, is_metric_size_based=is_metric)

    if is_reprojection:
        seg_centers, seg_normals, seg_covs, normal_covs, seg_sizes, seg_ids, segments = \
            get_segment_statistics(segments, verts, normals)

        mask_small_segments = seg_sizes < expected_segment_size
        mask_small_verts = mask_small_segments[segments]
        mask_for_reproj = ~mask_small_verts

        if is_split_large_segments:
            mask_too_large_segments = seg_sizes > 2 * expected_segment_size

            # split segments
            seg_centers_spl, seg_normals_spl, seg_covs_spl, normal_covs_spl, seg_sizes_spl = \
                split_large_segments(
                    seg_centers[mask_too_large_segments], seg_normals[mask_too_large_segments],
                    seg_covs[mask_too_large_segments], normal_covs[mask_too_large_segments],
                    seg_sizes[mask_too_large_segments])
            # compose new set of segments after splitting
            mask_not_too_large_segments = ~mask_too_large_segments
            seg_centers = np.concatenate(
                [seg_centers[mask_not_too_large_segments], seg_centers_spl])
            seg_normals = np.concatenate(
                [seg_normals[mask_not_too_large_segments], seg_normals_spl])
            seg_covs = np.concatenate(
                [seg_covs[mask_not_too_large_segments], seg_covs_spl])
            normal_covs = np.concatenate(
                [normal_covs[mask_not_too_large_segments], normal_covs_spl])
            seg_sizes = np.concatenate(
                [seg_sizes[mask_not_too_large_segments], seg_sizes_spl])
        segments_reproj = reproject_segments(verts[mask_for_reproj], normals[mask_for_reproj], seg_centers,
                                             seg_covs, seg_normals)
        segments[mask_for_reproj] = segments_reproj
        if is_merge_small_reprojection:
            segment_ids_old, segments = np.unique(
                segments, return_inverse=True)
            for i in range(0, len(segment_ids_old)):
                segment_mask = (segments == i)
                seg_sizes[i] = np.sum(segment_mask)
            segment_tree = np.arange(0, len(segment_ids_old))
            merge_small_segments(edges_sorted, segments, seg_sizes, small_segment_size,
                                 segment_tree=segment_tree)

            for i in range(0, len(segments)):
                segments[i] = get_vertex_root(segment_tree, segments[i])
    else:
        print('No reprojection')
    return segments


def segment_one_scene(dataset_type, scene_dir, mesh_name, **hyper_params):
    mesh_path = os.path.join(scene_dir, mesh_name)
    # verts [V, 3], labels [V,]
    mesh_data = read_ply(mesh_path)
    
    verts, labels, faces, normals, plydata = mesh_data["verts"], mesh_data["labels"], mesh_data["faces"], mesh_data["normals"], mesh_data["plydata"]
    # scannet raw labels are nyu40
    labels = np.asarray(labels).astype(np.int32)
    labels = nyu40_to_scannet20(labels)  # 0 to 20
    labels[labels > 20] = 0
    # save labels in scannet20 format 
    save_label_pth_path = os.path.join(scene_dir, "label.pth")
    if not os.path.exists(save_label_pth_path):
        torch.save(torch.from_numpy(labels).long(), save_label_pth_path)
        save_label_ply_path = os.path.join(scene_dir, "label.ply")
        vert_colors = vert_label_to_color(labels, color_encoding_scannet20)
        trimesh.Trimesh(verts, faces, vertex_colors=vert_colors).export(save_label_ply_path)

    if normals is None or hyper_params["o3d_normal"]:
        print("Using open3d normal computation!")
        normals = compute_normals_o3d(verts, faces)

    segment_suffix = hyper_params["segment_suffix"]
    output_dir = os.path.join(scene_dir, segment_suffix)
    os.makedirs(output_dir, exist_ok=True)
    
    # if os.path.exists(os.path.join(output_dir, "segments.pth")):
    #     return None, None, None, None, None, None, None
    
    # Do QPOS
    # [V,] save seg_ids
    segments = quasiplanar_segmentation(verts, normals, faces, **hyper_params)

    # compute oversegmentation error
    labels_major_vote,\
        lbl_err_mask_new, \
        oversegmentation_error,\
        avg_segsize,\
        min_seg_size,\
        max_seg_size,\
        n_small_1, \
        n_small_3,\
        n_small_5,\
        n_small_10,\
        n_seg, \
        avg_d_spat = oversegmentation_statistics(verts, normals, segments, labels)

    if labels_major_vote is not None:
        vert_colors = vert_label_to_color(labels_major_vote, color_encoding_scannet20)
        mesh_label_major_vote = trimesh.Trimesh(verts, faces, vertex_colors=vert_colors)
        mesh_label_major_vote.export(os.path.join(output_dir, "label_major_vote.ply"))
        # [V,] each vertex stores the label obtained via major vote
        torch.save(torch.from_numpy(labels_major_vote), os.path.join(output_dir, "label_major_vote.pth"))

    save_eval_ply(plydata, segments, output_dir, pref="classical")
    torch.save(torch.from_numpy(segments), os.path.join(output_dir, "segments.pth"))

    # visualize segments
    max_seg_id = np.max(segments)
    colour_map_seg = colour_map_using_hash(max_seg_id)
    colourize_instances(plydata, colour_map_seg, segments)
    plydata.write(os.path.join(output_dir, "segment.ply"))

    return oversegmentation_error, avg_segsize, min_seg_size, max_seg_size, (n_small_1, n_small_3, n_small_5, n_small_10), n_seg, avg_d_spat
