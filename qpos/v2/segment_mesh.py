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

from metric.iou import IoU3D
from dataio.utils import nyu40_to_scannet20, vert_label_to_color, color_encoding_scannet20
from config import get_scannet_root, get_scannet_test_root
from qpos.v2.geometric_instances import read_ply_scannet, segment_mesh, instances_floodfill
from plyfile import PlyData
import open3d as o3d
from tqdm import tqdm
import torch
import trimesh
import time
from sklearn.neighbors import KDTree
import numpy as np
import os.path
import sys
sys.path.append('/home/jingwen/vision/dev/supervoxel/build')
# import _slamcore_supervoxel as supervoxel


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
        r[i] = 255 * ((k[i] & 0x7fffffff) / float(0x7fffffff))
    return r


def colour_map_using_hash(max_seg):
    colour_map = np.zeros((max_seg+1, 3))
    for i in range(0, max_seg+1):
        colour_map[i] = hash_colour(i)
    return colour_map


def colour_map_using_hash_ids(ids):
    max_id = np.max(ids)
    colour_map = np.zeros((max_id+1, 3))
    for id in ids:
        colour_map[id] = hash_colour(id)
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


def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr /= np.tile(lens.reshape(-1, 1), (1, 3))
    return arr


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
    # same but organised as seg_ids
    seg_id_size = np.zeros(np.max(seg_ids) + 1)
    for i, sid in enumerate(seg_ids):
        mask = seg_labels == sid  # [V,]
        # if np.sum(mask) <= 1:
        #     continue

        # compute spatial dimension (diameter) of the segment_v1
        d_spat[i] = compute_diameter(verts[mask])
        # 3x3 spatial covariance matrix of the segment_v1
        spat_cov = np.cov(verts[mask].T)
        # sqrt of the trace of the covariance matrix
        trcov_spat[i] = np.sqrt(spat_cov.trace())
        # max chord length of the direction difference
        d_norm = compute_diameter(normals[mask])
        # max angle-difference in normals
        d_ang[i] = 2 * np.arcsin(np.min([d_norm / 2, 1.0])) * 180.0 / np.pi
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
    seg_sizes_srt = np.asarray(
        sorted(seg_sizes, reverse=True))  # descending order
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
            size_cdf[ind_100] * 100, size_cdf[ind_50] *
            100, size_cdf[ind_10] * 100
        )
    )

    if class_labels is not None:
        # Should respect_unknown be True or False? I guess False makes more sense?
        labels_majvote = label_segments(
            seg_labels, class_labels, respect_unknown=False)
        labels_err = np.abs(labels_majvote - class_labels)
        lbl_err_mask = labels_err > 0
        lbl_err_mask[class_labels == unknown_label] = 0
        oversegmentation_labelling_error = np.sum(
            lbl_err_mask) / len(lbl_err_mask) * 100
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

    # plt.show()

    return labels_majvote, lbl_err_mask, oversegmentation_labelling_error, np.mean(seg_sizes), np.min(seg_sizes), np.max(seg_sizes), n_seg, np.mean(d_spat)


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
    # (x_i - x_j)^t @ (x_i - x_j) = x_i^t @ x_i + x_j^t @ x_j - x_i^t @ x_j
    dists = vnh + vnv - 2 * vvt
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
    colour_map_seg = np.random.randint(
        0, size=(np.max(vert_lbls) + 1, 3), high=255)
    colourize_instances(plydata, colour_map_seg, vert_lbls)
    plydata.write(out_dir + "/" + pref + "_seg.ply")

    if lbl_err_mask is not None:
        colour_map_seg = np.zeros((2, 3))
        colour_map_seg[1] = np.asarray([255, 0, 0])
        colourize_instances(plydata, colour_map_seg,
                            lbl_err_mask.astype(np.uint32))
        plydata.write(out_dir + "/" + pref + "_err.ply")
        plydata.write(out_dir + "/" + pref + "_cov.ply")


def get_segment_statistics(segments, verts, normals, min_seg_size=0.01):
    """
    :param segments: [V,]
    :param verts: [V, 3]
    :param normals: [V, 3]
    :param min_seg_size: spatial dimension
    :return:
    """
    # compute segment_v1 stats
    segment_ids = np.unique(segments)
    seg_centers = np.zeros((len(segment_ids), 3))
    seg_normals = np.zeros((len(segment_ids), 3))
    seg_covs = np.zeros((len(segment_ids), 3, 3))
    normal_covs = np.zeros((len(segment_ids), 3, 3))
    seg_sizes = np.zeros((len(segment_ids)))
    seg_sizes_spat = np.zeros(len(segment_ids))
    for i, segment_id in enumerate(segment_ids):
        segment_mask = (segments == segment_id)
        if np.sum(segment_mask) < 2:  # n_verts
            continue
        seg_cov = np.cov(verts[segment_mask, :].transpose())
        seg_size_spat = np.sqrt(np.sum(np.diag(seg_cov)))
        if seg_size_spat < min_seg_size:  # spatial dim
            continue
        seg_centers[i] = np.mean(verts[segment_mask, :], axis=0)
        seg_covs[i] = np.cov(verts[segment_mask, :].transpose())
        normal_covs[i] = np.cov(normals[segment_mask, :].transpose())
        mean_normal = np.mean(normals[segment_mask, :], axis=0)
        seg_normals[i] = mean_normal / np.linalg.norm(mean_normal)
        seg_sizes[i] = np.sum(segment_mask)
        seg_sizes_spat[i] = seg_size_spat
    good_seg_mask = seg_sizes > 0
    seg_centers = seg_centers[good_seg_mask]
    seg_normals = seg_normals[good_seg_mask]
    seg_covs = seg_covs[good_seg_mask]
    normal_covs = normal_covs[good_seg_mask]

    print('avg segment size {}'.format(np.mean(seg_sizes)))
    print('avg segment spatial size {}'.format(np.mean(seg_sizes_spat)))
    # compute segment_v1 size histogram
    # plt.hist(seg_sizes, bins='auto')
    # plt.title("Histogram with 'auto' bins")
    # plt.show()

    return seg_centers, seg_normals, seg_covs, normal_covs


def reproject_segments(verts, normals, seg_centers, seg_covs, seg_normals, seg_ids=None):
    """
    :param verts: [V, 3]
    :param normals: [V, 3]
    :param seg_centers: [N_seg, 3]
    :param seg_covs: [N_seg, 3, 3]
    :param seg_normals: [N_seg, 3]
    :param normal_covs: [N_seg, 3, 3]
    :return:
    """
    segments_new = np.ones((verts.shape[0]), dtype=np.int32) * (-1)
    if seg_ids is None:
        seg_ids = np.arange(0, seg_centers.shape[0], dtype=int)
    tree = KDTree(seg_centers[seg_ids])
    num_neighbours = np.min([5, len(seg_ids)])
    normal_cost_weight = 5.0
    # For each vertex find its k-nearest segments
    nn_inds = tree.query(verts, k=num_neighbours,
                         return_distance=False, sort_results=True)                      
    # for vi, seg_ind in enumerate(nn_inds):
    #     num_neighbours = len(seg_ind)
    #     costs = np.zeros(num_neighbours)
    #     for ni in range(0, len(seg_ind)):
    #         curr_seg = seg_ids[seg_ind[ni]]
    #         dp = verts[vi] - seg_centers[curr_seg]
    #         costs[ni] = dp.reshape(
    #             1, 3) @ np.linalg.inv(seg_covs[curr_seg] + 1e-6*np.eye(3)) @ dp.reshape(3, 1)
    #         # dn = normals[vi] - seg_normals[curr_seg]
    #         # normal_cost = dn.reshape(1, 3) @ np.linalg.inv(normal_covs[curr_seg]) @ dn.reshape(3, 1)
    #         normal_cost = 1.0 - \
    #                       np.abs(np.sum(normals[vi] * seg_normals[curr_seg]))
    #         costs[ni] += normal_cost_weight * normal_cost
    #     # label the vertex with the segment_v1 with smallest cost
    #     min_fit = np.argmin(costs)
    #     segments_new[vi] = seg_ids[seg_ind[min_fit]]

    curr_seg = seg_ids[nn_inds] # n_verts x num_neighbours
    dp = verts.reshape(-1, 1, 3) - seg_centers[curr_seg, :] # n_verts x num_neighbours x 3
    seg_covs_inv = np.linalg.inv(seg_covs + 1e-6*np.eye(3).reshape(1, 3, 3)) # n_verts x 3 x 3
    curr_seg_inv_covs = seg_covs_inv[curr_seg, :, :] # n_verts x num_neighbours x 3 x 3
    left_mult_cost = np.einsum('ijk,ijkl->ijl',dp, curr_seg_inv_covs)
    cost = np.einsum('ijk,ijk->ij',left_mult_cost, dp) # n_verts x num_neighbours
    seg_normals_sampled = seg_normals[curr_seg, :] # n_verts x num_neighbours x 3
    cost += normal_cost_weight * (1.0 - np.abs(np.sum(normals.reshape(-1, 1, 3) * seg_normals_sampled, 2)))
    min_cost_inds = np.argmin(cost, 1)
    min_cost_vert_inds = nn_inds[np.arange(0, nn_inds.shape[0]), min_cost_inds] # n_verts
    # print(min_cost_vert_inds.shape)
    segments_new = seg_ids[min_cost_vert_inds]
    # print('fast; accurate? ' + str(np.linalg.norm(segments_new_est-segments_new)))

    return segments_new


def supervoxel_segmentation(verts, resolution=0.05):
    segments = supervoxel.segment(verts, resolution)
    return np.asarray(segments)


def quasiplanar_segmentation(verts, normals, faces,
                             is_reprojection=True,
                             expected_segment_size=0.1,
                             small_segment_size=0.02,
                             max_weight=1.0,
                             w_shape=0.0,
                             w_normal=0.0,
                             **kwargs):
    """
    :param verts:
    :param normals:
    :param faces:
    :param is_reprojection:
    :param k_segmentation: larger constant - larger segments
    :param small_segment_size: small segment_v1 size (number of vertices), we merge segments if the segments are smaller than that
    :param max_segment_size: expected max segment_v1 size (number of vertices), sometimes segments can be bigger due to small segments merging
    :return:
    """

    # EGS segmentation
    # fine_params = [k_segmentation, small_segment_size, expected_segment_size, max_weight, w_shape]
    fine_params = [small_segment_size,
                   expected_segment_size, w_shape, w_normal]
    vertex_tree = segment_mesh(faces, normals, verts, fine_params)
    segments = instances_floodfill(vertex_tree)  # [V,]

    # segment_v1 statistics (mean vertex, normal, covariances)
    # segment_v1 center: [N_seg, 3]
    # segment_v1 normal: [N_seg, 3]
    seg_centers, seg_normals, seg_covs, normal_covs = \
        get_segment_statistics(segments, verts, normals,
                               min_seg_size=small_segment_size)

    if is_reprojection:
        segments = reproject_segments(
            verts, normals, seg_centers, seg_covs, seg_normals)

    return segments


def segment_one_scene(dataset_type, scene, mode="qp", **hyper_params):
    if dataset_type == "scannet":
        scene_dir = os.path.join(get_scannet_root(), scene)
        mesh_path = os.path.join(
            scene_dir, "{}_vh_clean_2.labels.ply".format(scene))
        read_ply = read_ply_scannet
        # [V, 3], [V,]
        verts, labels, instances, faces, normals, plydata = read_ply(mesh_path)
        labels = np.asarray(labels).astype(np.int32)
        labels = nyu40_to_scannet20(labels)  # 0 to 20
        labels[labels > 20] = 0
    elif dataset_type == "scannet_test":
        scene_dir = os.path.join(get_scannet_test_root(), scene)
        mesh_path = os.path.join(scene_dir, "{}_vh_clean_2.ply".format(scene))
        read_ply = read_ply_scannet
        # [V, 3], [V,]
        verts, labels, instances, faces, normals, plydata = read_ply(mesh_path)
    elif dataset_type == "scenenn":
        scene_dir = os.path.join(get_scenenn_semantic_root(), scene)
        mesh_verts_data = torch.load(os.path.join(scene_dir, "mesh_verts.pth"))
        verts = mesh_verts_data["verts"].cpu().numpy()
        normals = mesh_verts_data["normals"].cpu().numpy()
        faces = torch.load(os.path.join(
            scene_dir, "mesh_faces.pth")).cpu().numpy()
        labels = torch.load(os.path.join(
            scene_dir, "{}_label_scannet20.pth".format(scene))).cpu().numpy()
        ply_path = os.path.join(scene_dir, "{}.ply".format(scene))
        with open(ply_path, "rb") as f:
            plydata = PlyData.read(f)
            plydata.elements[0]["x"] = verts[:, 0]
            plydata.elements[0]["y"] = verts[:, 1]
            plydata.elements[0]["z"] = verts[:, 2]
            plydata.elements[0]["nx"] = normals[:, 0]
            plydata.elements[0]["ny"] = normals[:, 1]
            plydata.elements[0]["nz"] = normals[:, 2]
    else:
        raise NotImplementedError

    if normals is None:
        if hyper_params["o3d_normal"]:
            print("Using open3d normal computation!")
            normals = compute_normals_o3d(verts, faces)
        else:
            normals = compute_normals(verts, faces)

    if mode == "qp":
        output_dir = os.path.join(scene_dir, "segments/qp/spat_{}_{}_{}".format(hyper_params["is_reprojection"],
                                                                                hyper_params["small_segment_size"],
                                                                                hyper_params["expected_segment_size"]))
        if hyper_params["o3d_normal"]:
            output_dir += "_o3d"

        if hyper_params["debug"]:
            output_dir += "_debug"

        # [V,] storing seg_ids
        segments = quasiplanar_segmentation(
            verts, normals, faces, **hyper_params)
    elif mode == "sv":
        output_dir = os.path.join(
            scene_dir, "segments/sv/{}".format(hyper_params["resolution"]))
        segments = supervoxel_segmentation(verts, **hyper_params)
    else:
        raise NotImplementedError

    # if os.path.exists(os.path.join(output_dir, "classical_seg.ply")):
    #     print("Already done!")
    #     return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # compute oversegmentation error
    labels_major_vote, lbl_err_mask_new, oversegmentation_error, avg_segsize, min_seg_size, max_seg_size, n_seg, avg_d_spat = oversegmentation_statistics(
        verts, normals, segments, labels
    )

    if labels_major_vote is not None:
        vert_colors = vert_label_to_color(
            labels_major_vote, color_encoding_scannet20)
        mesh_label_major_vote = trimesh.Trimesh(
            verts, faces, vertex_colors=vert_colors)
        mesh_label_major_vote.export(os.path.join(
            output_dir, "label_major_vote.ply"))
        # [V,] each vertex stores the label obtained via major vote
        torch.save(torch.from_numpy(labels_major_vote),
                   os.path.join(output_dir, "label_major_vote.pth"))

    save_eval_ply(plydata, segments, output_dir, pref="classical")
    torch.save(torch.from_numpy(segments),
               os.path.join(output_dir, "segments.pth"))

    # visualize segments
    max_seg_id = np.max(segments)
    colour_map_seg = colour_map_using_hash(max_seg_id)
    colourize_instances(plydata, colour_map_seg, segments)
    plydata.write(os.path.join(output_dir, "segment.ply"))

    return oversegmentation_error, avg_segsize, min_seg_size, max_seg_size, n_seg, avg_d_spat


def segment_meshes(hyper_qp, scene_list):
    modes = ["qp", "sv"]

    errors, avg_seg_sizes, n_segs, avg_d_spats = dict(), dict(), dict(), dict()
    for scene in tqdm(scene_list):
        err, avg_seg_size, n_seg, avg_d_spat = segment_one_scene(
            SCANNET_ROOT, scene, mode=modes[0], **hyper_qp)
        errors[scene] = err
        avg_seg_sizes[scene] = avg_seg_size
        n_segs[scene] = n_seg
        avg_d_spats[scene] = avg_d_spat

    result_filename = "segment_{}_{}_{}.txt".format(hyper_qp["is_reprojection"],
                                                    hyper_qp["small_segment_size"],
                                                    hyper_qp["expected_segment_size"])
    with open(result_filename, "w") as f:
        f.write("-------------Evaluation Result--------------\n")
        for scene in errors:
            f.write("{}: {:.2f}%, {:.4f}, {}\n".format(
                scene, errors[scene], avg_seg_sizes[scene], n_segs[scene]))

        error_list = np.array(list(errors.values()))
        error_mean = error_list.mean()
        error_std = error_list.std()
        error_max = error_list.max()
        error_min = error_list.min()
        f.write("Over-segmentation stats:\n")
        f.write("mean: {:.2f}\n".format(error_mean))
        f.write("std: {:.2f}\n".format(error_std))
        f.write("max: {:.2f}\n".format(error_max))
        f.write("min: {:.2f}\n".format(error_min))

        size_list = np.array(list(avg_seg_sizes.values()))
        size_mean = size_list.mean()
        size_std = size_list.std()
        size_max = size_list.max()
        size_min = size_list.min()
        f.write("Average segment_v1 size stats:\n")
        f.write("mean: {:.4f}\n".format(size_mean))
        f.write("std: {:.4f}\n".format(size_std))
        f.write("max: {:.4f}\n".format(size_max))
        f.write("min: {:.4f}\n".format(size_min))

        n_seg_list = np.array(list(n_segs.values()))
        n_seg_mean = n_seg_list.mean()
        n_seg_std = n_seg_list.std()
        n_seg_max = n_seg_list.max()
        n_seg_min = n_seg_list.min()
        f.write("Number of segments stats:\n")
        f.write("mean: {:.1f}\n".format(n_seg_mean))
        f.write("std: {:.1f}\n".format(n_seg_std))
        f.write("max: {}\n".format(n_seg_max))
        f.write("min: {}\n".format(n_seg_min))

        d_spat_list = np.array(list(avg_d_spats.values()))
        d_spat_mean = d_spat_list.mean()
        d_spat_std = d_spat_list.std()
        d_spat_max = d_spat_list.max()
        d_spat_min = d_spat_list.min()
        f.write("Average segment_v1 size stats:\n")
        f.write("mean: {:.4f}\n".format(d_spat_mean))
        f.write("std: {:.4f}\n".format(d_spat_std))
        f.write("max: {:.4f}\n".format(d_spat_max))
        f.write("min: {:.4f}\n".format(d_spat_min))

    # segment_one_scene(SCANNET_ROOT, scene, mode=modes[1], **hyper_sv)


def evaluate_iou(hyper_params, scene_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scannet_root = get_scannet_root()
    metric = IoU3D(21, ignore_index=0)
    for scene in tqdm(scene_list):
        segment_dir = os.path.join(scannet_root, "{}/segments/qp/{}_{}_{}".format(scene,
                                                                                  hyper_params["is_reprojection"],
                                                                                  hyper_params["small_segment_size"],
                                                                                  hyper_params["expected_segment_size"]))
        if hyper_qp["o3d_normal"]:
            segment_dir += "_o3d"
        label_pred = torch.load(os.path.join(
            segment_dir, "label_major_vote.pth"), map_location=device)
        label_gt = torch.load(os.path.join(
            scannet_root, "{}/label.pth".format(scene)), map_location=device)
        metric.add(label_pred, label_gt)

    iou, miou = metric.value()
    filename = "IoU_segment_{}_{}_{}".format(hyper_params["is_reprojection"],
                                             hyper_params["small_segment_size"],
                                             hyper_params["expected_segment_size"])
    if hyper_qp["o3d_normal"]:
        filename += "_o3d"
    result_filename = filename + ".txt"
    with open(result_filename, "w") as f:
        f.write("-------------IoU Result--------------\n")
        for key, class_iou in zip(color_encoding_scannet20.keys(), iou):
            f.write("{0}: {1:.4f}\n".format(key, class_iou))
        f.write("Mean IoU: {}\n".format(miou))
        f.write("Mean IoU (ignore unlabelled): {}".format(iou[1:].mean()))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_reprojection", type=bool, default=True)
    parser.add_argument("--small_segment_size", type=float, default=0.02)
    parser.add_argument("--expected_segment_size", type=float, default=0.1)
    args = parser.parse_args()
    # Hyper-parameters for quasi-planar segment
    hyper_qp = {
        "is_reprojection": args.is_reprojection,
        "small_segment_size": args.small_segment_size,  # 0.02
        "expected_segment_size": args.expected_segment_size,  # 0.1,
        "o3d_normal": True
    }

    dataset_types = ["scannet", "scenenn"]
    dataset_type = dataset_types[1]
    scene = "045"
    segment_one_scene(dataset_type, scene, mode="qp", **hyper_qp)