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

import numpy as np
import time


def normalize_v3(arr):
    """Normalize a numpy array of 3 component vectors shape=(n,3)"""
    lens = np.sqrt(
        arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2 + np.finfo(float).eps
    )
    arr /= np.tile(lens.reshape(-1, 1), (1, 3))
    return arr


"""
edge_ids: n x 2 array with columns: vertex 1 id, vertex 2 id
rdge_weights: array of size n with the edge weights
vertices are enumerated with non-negative integers continuously
"""


def segment_graph_normals(
        edge_ids, pos, normals, max_norm_dist, max_pos_dist, join_pos_thr=0
):
    """
    Segment the mesh using bounded normal-based segmentation
    :param edge_ids: edges array with vertex ids
    :param pos: vertex positions
    :param normals: vertex normals
    :param max_norm_dist: threshold for normal distance
    :param max_pos_dist: positional threshold
    :param join_pos_thr: threshold for small segment_v1 merging
    :return: vertex->segment_v1 map, segment_v1 sizes, ids for sorting edges
    """
    vert_num = np.max(edge_ids.reshape(-1)) + 1
    # array of vert_num, where we store the segment_v1 id
    vert_tree = np.arange(0, vert_num, dtype=np.uint32)
    # the vertex count for each vertex and the largest weight for each segment_v1
    vert_cnt = np.ones(vert_num, dtype=np.uint32)
    n_vert = normals.shape[0]
    # edge_weights = 1.0 / max_norm_dist * np.linalg.norm(
    #     norm_diffs, axis=1)
    dot_prod = np.abs(np.sum(normals[edge_ids[:, 0]] * normals[edge_ids[:, 1]], axis=1))
    edge_weights = 1.0 / max_norm_dist * dot_prod
    pos_diffs = pos[edge_ids[:, 0]] - pos[edge_ids[:, 1]]
    if max_pos_dist > 0:
        edge_weights += 1.0 / max_pos_dist * np.linalg.norm(pos_diffs, axis=1)
    normal_covs = np.ones((n_vert, 3, 3)) * 0.0001
    pos_covs = np.ones((n_vert, 3, 3)) * (0.01 * 0.01) # 1 cm
    edge_sort_ids = np.argsort(edge_weights)
    for e_id in edge_sort_ids:
        w = edge_weights[e_id]
        v1_id, v2_id = edge_ids[e_id]
        s1_id = get_vertex_root(vert_tree, v1_id)
        s2_id = get_vertex_root(vert_tree, v2_id)
        norm_cov_1 = np.sqrt(np.diagonal(normal_covs[s1_id]).max())
        pos_cov_1 = np.sqrt(np.diagonal(pos_covs[s1_id]).max())
        thr1 = 2 - norm_cov_1 / max_norm_dist - pos_cov_1 / max_pos_dist
        norm_cov_2 = np.sqrt(np.diagonal(normal_covs[s2_id]).max())
        pos_cov_2 = np.sqrt(np.diagonal(pos_covs[s2_id]).max())
        thr2 = 2 - norm_cov_2 / max_norm_dist - pos_cov_2 / max_pos_dist
        merge_thr = np.min([thr1, thr2])
        if s1_id != s2_id and w < merge_thr:
            # merge s1_id to s2_id
            merge_with_normals(
                s1_id, s2_id, vert_tree, normals, normal_covs, vert_cnt, pos, pos_covs
            )

    # in case we want to merge small segments...
    for e_id in edge_sort_ids:
        v1_id, v2_id = edge_ids[e_id]
        s1_id = get_vertex_root(vert_tree, v1_id)
        s2_id = get_vertex_root(vert_tree, v2_id)
        if s1_id != s2_id:
            sigma1 = np.sqrt(np.diagonal(pos_covs[s1_id]).max())
            sigma2 = np.sqrt(np.diagonal(pos_covs[s2_id]).max())
            if sigma1 < join_pos_thr or sigma2 < join_pos_thr:
                merge_with_normals(
                    s1_id,
                    s2_id,
                    vert_tree,
                    normals,
                    normal_covs,
                    vert_cnt,
                    pos,
                    pos_covs,
                )

    for v_id in range(0, vert_num):
        get_vertex_root(vert_tree, v_id)
    return vert_tree, vert_cnt, edge_sort_ids, pos, pos_covs, normals


def segment_graph(edge_ids, edge_weights, c, max_num_vertices, max_weight):
    """
    Efficient graph segmentation by Felszenwalb and Huttenlocher for meshes

    :param edge_ids: edges array with vertex ids [3F, 2]
    :param edge_weights: edge weights, normally 1 - dot product of normals
    :param c: algorithm parameter, k_segmentation
    :return: vertex->segment_v1 map, segment_v1 sizes, ids for sorting edges
    """
    vert_num = np.max(edge_ids.reshape(-1)) + 1
    # array of vert_num, where we store the segment_v1 id
    vert_tree = np.arange(0, vert_num, dtype=np.uint32)
    # the vertex count for each vertex and the largest weight for each segment_v1
    vert_cnt = np.ones(vert_num, dtype=np.uint32)
    # the average normal
    w_last = np.zeros(vert_num)
    edge_sort_ids = np.argsort(edge_weights)
    # ascending order over weight value
    for e_id in edge_sort_ids:
        w = edge_weights[e_id]
        if w > max_weight:
            continue
        v1_id, v2_id = edge_ids[e_id]
        s1_id = get_vertex_root(vert_tree, v1_id)
        s2_id = get_vertex_root(vert_tree, v2_id)
        if s1_id != s2_id and w < np.min(
                [w_last[s1_id] + c / vert_cnt[s1_id], w_last[s2_id] + c / vert_cnt[s2_id]]) and \
                vert_cnt[s2_id] + vert_cnt[s1_id] < max_num_vertices:
            # merge s1_id to s2_id
            vert_tree[s1_id] = s2_id
            vert_cnt[s2_id] += vert_cnt[s1_id]
            w_last[s2_id] = w
    return vert_tree, vert_cnt, edge_sort_ids


def get_size_from_covariance(cov):
    return np.linalg.norm(np.sqrt(np.diag(cov)))


def compute_energy(max_segment_size, cov):
    # size_term = np.abs(cnt - max_segment_size)

    size_estimate = get_size_from_covariance(cov)
    # print('{}-{}'.format(size_estimate, max_segment_size))
    size_term = np.abs(size_estimate - max_segment_size)

    # shape_term = sorted_vals[2]/sorted_vals[1]
    # return size_term + weight_shape * shape_term
    return size_term


def get_normal_sigma(mean_normal):
    R = np.linalg.norm(mean_normal)
    p = 3
    if np.abs(R - 1.0) < 1e-5:
        return 1e-5
    kappa_est = R * (p - R * R) / ( 1 - R * R)
    if kappa_est < 1e-5:
        print('error!')
        print(R)
        print(kappa_est)
        exit()
    sigma_est = 1.0 / np.sqrt(kappa_est)
    return sigma_est


def compute_energy_normal_adaptive(mean_normal, max_segment_size, cov, weight_shape,
                                   weight_normal):
    # size_term = np.abs(cnt - max_segment_size)
    # vals, vecs = np.linalg.eigh(cov)
    cov_diag = np.diag(cov)

    # sorted_vals = sorted(vals)
    sorted_vals = sorted(cov_diag)
    size_estimate = np.sqrt(sorted_vals[2] * sorted_vals[1])
    normal_sigma = get_normal_sigma(mean_normal)
    normal_term = normal_sigma * normal_sigma

    # print('s {} n {}'.format(np.abs(size_estimate - max_segment_size), normal_term))
    full_size_estimate = size_estimate + weight_normal * normal_term
    size_term = np.abs(full_size_estimate - max_segment_size)

    shape_term = sorted_vals[2] / sorted_vals[1]
    return size_term + weight_shape * shape_term


def check_energy(merged_cov, cov_1, cov_2, max_segment_size):
    # d_1 = np.abs(cnt_1-max_segment_size)
    # d_2 = np.abs(cnt_2-max_segment_size)
    # d_m = np.abs(merged_cnt - max_segment_size)

    e_1 = compute_energy(max_segment_size, cov_1)
    e_2 = compute_energy(max_segment_size, cov_2)
    e_m = compute_energy(max_segment_size, merged_cov)
    # e_1 = compute_energy_normal_adaptive(normal_1, max_segment_size, cov_1,
    # weight_shape, weight_normal)
    # e_2 = compute_energy_normal_adaptive(normal_2, max_segment_size, cov_2,
    #     weight_shape, weight_normal)
    # e_m = compute_energy_normal_adaptive(merged_normal, max_segment_size, merged_cov,
    #     weight_shape, weight_normal)
    # e_m = compute_energy(merged_cnt, max_segment_size, merged_cov, weight_shape)

    if 0.5 * (e_1 + e_2) < e_m:
        return False
    else:
        return True


def try_merge_using_metric_size(vert_tree, vert_cnt, seg_means, seg_covs, seg_normals,
                                max_num_vertices, s1_id, s2_id):
    merged_mean, merged_cov = merge_mean_cov_est(
        seg_means[s1_id], seg_covs[s1_id], vert_cnt[s1_id],
        seg_means[s2_id], seg_covs[s2_id], vert_cnt[s2_id])
    merged_cov = np.diag(np.diag(merged_cov))
    merged_cnt = vert_cnt[s1_id] + vert_cnt[s2_id]
    w_1 = vert_cnt[s1_id] / merged_cnt
    w_2 = 1 - w_1
    merged_normal = w_1 * seg_normals[s1_id] + w_2 * seg_normals[s2_id]
    if check_energy(merged_cov, seg_covs[s1_id], seg_covs[s2_id],
                    max_num_vertices):
        vert_tree[s1_id] = s2_id
        vert_cnt[s2_id] = merged_cnt
        seg_means[s2_id] = merged_mean
        seg_covs[s2_id] = merged_cov
        seg_normals[s2_id] = merged_normal


def try_merge_using_vertex_count(vert_tree, vert_cnt, seg_means, seg_covs, seg_normals,
                                 max_num_vertices, s1_id, s2_id):
    merged_cnt = vert_cnt[s1_id] + vert_cnt[s2_id]
    merged_cost = np.abs(merged_cnt-max_num_vertices)
    cost_1 = np.abs(vert_cnt[s1_id] - max_num_vertices)
    cost_2 = np.abs(vert_cnt[s2_id] - max_num_vertices)
    if (merged_cost < 0.5 * (cost_1 + cost_2)):
        merged_mean, merged_cov = merge_mean_cov_est(
            seg_means[s1_id], seg_covs[s1_id], vert_cnt[s1_id],
            seg_means[s2_id], seg_covs[s2_id], vert_cnt[s2_id])
        merged_cov = np.diag(np.diag(merged_cov))
        w_1 = vert_cnt[s1_id] / merged_cnt
        w_2 = 1 - w_1
        merged_normal = w_1 * \
                        seg_normals[s1_id] + w_2 * seg_normals[s2_id]
        vert_tree[s1_id] = s2_id
        vert_cnt[s2_id] = merged_cnt
        seg_means[s2_id] = merged_mean
        seg_covs[s2_id] = merged_cov
        seg_normals[s2_id] = merged_normal


def merge_small_segments(edge_ids, segments, vert_cnt, small_size, seg_means=None, seg_covs=None, seg_normals=None, segment_tree=None):
    if segment_tree is None:
        segment_tree = segments
    for e in edge_ids:
        v1_id, v2_id = e
        s1_id = get_vertex_root(segment_tree, segments[v1_id])
        s2_id = get_vertex_root(segment_tree, segments[v2_id])
        if s1_id != s2_id:
            # try merge
            if (vert_cnt[s1_id] < small_size) or (vert_cnt[s2_id] < small_size):
                segment_tree[s1_id] = s2_id
                merged_cnt = vert_cnt[s1_id] + vert_cnt[s2_id]
                vert_cnt[s2_id] = merged_cnt
                if seg_means is not None:
                    merged_mean, merged_cov = merge_mean_cov_est(
                        seg_means[s1_id], seg_covs[s1_id], vert_cnt[s1_id],
                        seg_means[s2_id], seg_covs[s2_id], vert_cnt[s2_id])
                    merged_cov = np.diag(np.diag(merged_cov))
                    w_1 = vert_cnt[s1_id] / merged_cnt
                    w_2 = 1 - w_1
                    merged_normal = w_1 * seg_normals[s1_id] + w_2 * seg_normals[s2_id]
                    seg_means[s2_id] = merged_mean
                    seg_covs[s2_id] = merged_cov
                    seg_normals[s2_id] = merged_normal


def segment_graph_energy(verts, normals, edge_ids, edge_weights,
                         max_num_vertices, is_metric_size_based=True,
                         is_merge_small=False, small_size=0):
    """
    Efficient graph segmentation by Felszenwalb and Huttenlocher for meshes

    :param edge_ids: edges array with vertex ids
    :param edge_weights: edge weights, normally 1 - dot product of normals
    :param c: algorithm parameter
    :return: vertex->segment map, segment sizes, ids for sorting edges
    """
    # vert_num = np.max(edge_ids.reshape(-1)) + 1
    vert_num = len(verts)
    # array of vert_num, where we store the segment id
    vert_tree = np.arange(0, vert_num, dtype=np.uint32)
    # the vertex count for each vertex and the largest weight for each segment
    vert_cnt = np.ones(vert_num, dtype=np.uint32)
    # segment means
    seg_means = np.copy(verts)
    # segment covariances
    init_cov = np.eye(3) * 0.005 * 0.005  # 5mm sigma
    seg_covs = np.tile(init_cov.reshape((1, 3, 3)), (vert_num, 1, 1))
    seg_normals = np.copy(normals)
    edge_sort_ids = np.argsort(edge_weights)
    for e_id in edge_sort_ids:
        v1_id, v2_id = edge_ids[e_id]
        s1_id = get_vertex_root(vert_tree, v1_id)
        s2_id = get_vertex_root(vert_tree, v2_id)
        if s1_id != s2_id:
            # try merge
            if is_metric_size_based:
                try_merge_using_metric_size(vert_tree, vert_cnt, seg_means, seg_covs, seg_normals,
                                            max_num_vertices, s1_id, s2_id)
            else:
                try_merge_using_vertex_count(vert_tree, vert_cnt, seg_means, seg_covs, seg_normals,
                                             max_num_vertices, s1_id, s2_id)

    if is_merge_small:
        merge_small_segments(edge_ids[edge_sort_ids], vert_tree, vert_cnt, small_size,
                             seg_means, seg_covs, seg_normals)

    # fix vertex_tree to be a direct vertex-to-segment mapping
    for i in range(0, len(vert_tree)):
        get_vertex_root(vert_tree, i)
    return vert_tree, vert_cnt, edge_sort_ids


def print_cov(cov_1, lbl):
    print(lbl)
    vals, vecs = np.linalg.eigh(cov_1)
    print(sorted(vals))


def merge_mean_cov_est(mu_1, cov_1, n_1, mu_2, cov_2, n_2):
    """
    Recursive merge of two mean-covariance estimates
    :param mu_1: first sub-sample mean
    :param cov_1: first sub-sample covariance
    :param n_1: number of samples in a first sub-sample
    :param mu_2: second sub-sample mean
    :param cov_2: second sub-sample covariance
    :param n_2: number of samples in a second sub-sample
    :return: mean and covariance
    """
    if (n_1 == 1) or (n_2 == 1):
        n_1 *= 10
        n_2 *= 10
    n = n_1 + n_2
    mu_res = n_1 / n * mu_1 + n_2 / n * mu_2
    mu_1 = mu_1.reshape(-1, 1)
    mu_2 = mu_2.reshape(-1, 1)
    cov_res = (
            (n_1 - 1) / (n - 1) * cov_1
            + (n_2 - 1) / (n - 1) * cov_2
            + n_1 * n_2 / n / (n - 1) * (mu_1 - mu_2) @ (mu_1 - mu_2).transpose()
    )
    return mu_res, cov_res


def merge_mean_cov_est_weight(mu_1, cov_1, w_1, mu_2, cov_2, w_2):
    """
    Recursive merge of two mean-covariance estimates
    :param mu_1: first sub-sample mean
    :param cov_1: first sub-sample covariance
    :param mu_2: second sub-sample mean
    :param cov_2: second sub-sample covariance
    :return: mean and covariance
    """
    w_1 = w_1 / (w_1 + w_2)
    w_2 = 1 - w_1
    mu_res = w_1 * mu_1 + w_2 * mu_2
    mu_1 = mu_1.reshape(-1, 1)
    mu_2 = mu_2.reshape(-1, 1)
    dmu = mu_2 - mu_1
    cov_res = (
            w_1 * cov_1
            + w_2 * cov_2
            + (w_1 * w_2) * dmu @ dmu.transpose()
    )
    return mu_res, cov_res


def merge_with_normals(
        s1_id, s2_id, vert_tree, normals, normal_covs, vert_cnt, pos, pos_covs
):
    """
    Merge two segments

    :param s1_id: id of a 1st segment_v1
    :param s2_id: id of a 2nd segment_v1
    :param vert_tree: vertex->segment_v1 map
    :param normals: normals
    :param normal_covs: normal covariances
    :param vert_cnt: segment_v1 sizes
    :param pos: segment_v1 spatial centers
    :param pos_covs: segment_v1 spatial covariances
    """
    vert_tree[s1_id] = s2_id
    mu_res, cov_res = merge_mean_cov_est(
        normals[s1_id],
        normal_covs[s1_id],
        vert_cnt[s1_id],
        normals[s2_id],
        normal_covs[s2_id],
        vert_cnt[s2_id],
    )

    normals[s2_id] = mu_res
    normal_covs[s2_id] = cov_res
    pos_res, cov_res = merge_mean_cov_est(
        pos[s1_id],
        pos_covs[s1_id],
        vert_cnt[s1_id],
        pos[s2_id],
        pos_covs[s2_id],
        vert_cnt[s2_id],
    )
    pos[s2_id] = pos_res
    pos_covs[s2_id] = cov_res
    vert_cnt[s2_id] += vert_cnt[s1_id]


def get_vertex_root(vert_tree, v_id):
    init_v_id = v_id
    while vert_tree[v_id] != v_id:
        v_id = vert_tree[v_id]
    vert_tree[init_v_id] = v_id
    return v_id


def segment_mesh(faces, normals, verts, expected_size, small_size, is_merge_small, is_metric_size_based):
    """
    Segment the mesh

    :param faces: n_face x 3 array of the vertex indices defining the faces
    :param normals: n_vert x 3 array of vertex normals
    :param verts: n_vert x 3 array of vertices
    :param params of the segmentation algorithm
    :return: an array of segment_v1 indices for each vertex
    """
    print("Start segmentation")
    t_start = time.time()
    # adjust normals to unit norm
    normals = normalize_v3(normals)
    # compute edge weights
    edge_ids = np.concatenate([faces[:, 0:2], faces[:, 1:3], faces[:, [0, 2]]], axis=0)  # [3F, 2]
    edge_weights = 1 - np.sum(
        normals[edge_ids[:, 0]] * normals[edge_ids[:, 1]], axis=1
    )  # very small?
    dp = verts[edge_ids[:, 1]] - verts[edge_ids[:, 0]]  # [3F, 3]
    conv_signs = (np.sum(dp * normals[edge_ids[:, 1]], axis=1) > 0)
    edge_weights[conv_signs] = edge_weights[conv_signs] * edge_weights[conv_signs]
    # run segmentation
    vertex_tree, vertex_cnt, edge_sort_ids = segment_graph_energy(
        verts, normals, edge_ids, edge_weights, expected_size,
        small_size=small_size, is_merge_small=is_merge_small,
        is_metric_size_based=is_metric_size_based)
    # make segment labels from 0 to max(vertex_tree)
    segment_ids, segments = np.unique(vertex_tree, return_inverse=True)

    print("Segmentation took {} sec.".format(time.time() - t_start))
    return segments, edge_ids[edge_sort_ids]
