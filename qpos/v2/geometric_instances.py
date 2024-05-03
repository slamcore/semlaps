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
import argparse
import numpy as np
import time
from plyfile import PlyData, PlyElement

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


def compute_energy(cnt, max_segment_size, cov, weight_shape):
    # size_term = np.abs(cnt - max_segment_size)
    cov_diag = np.diag(cov)
    sorted_vals = sorted(cov_diag)

    size_estimate = np.linalg.norm(np.sqrt(cov_diag))
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

    shape_term = sorted_vals[2]/sorted_vals[1]
    return size_term + weight_shape * shape_term


def check_energy(merged_cov, cov_1, cov_2, merged_cnt, cnt_1, cnt_2,
                 merged_normal, normal_1, normal_2, max_segment_size, weight_shape, weight_normal):
    d_1 = np.abs(cnt_1-max_segment_size)
    d_2 = np.abs(cnt_2-max_segment_size)
    d_m = np.abs(merged_cnt - max_segment_size)

    e_1 = compute_energy(cnt_1, max_segment_size, cov_1, weight_shape)
    e_2 = compute_energy(cnt_2, max_segment_size, cov_2, weight_shape)
    e_m = compute_energy(merged_cnt, max_segment_size, merged_cov, weight_shape)
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


def segment_graph_energy(verts, normals, edge_ids, edge_weights,
                         max_num_vertices, max_weight, weight_shape, weight_normal):
    """
    Efficient graph segmentation by Felszenwalb and Huttenlocher for meshes

    :param edge_ids: edges array with vertex ids
    :param edge_weights: edge weights, normally 1 - dot product of normals
    :param c: algorithm parameter
    :return: vertex->segment map, segment sizes, ids for sorting edges
    """
    # TODO: in very rare cases, not all vertices are rederenced...
    # vert_num = np.max(edge_ids.reshape(-1)) + 1
    vert_num = len(verts)
    # array of vert_num, where we store the segment id
    vert_tree = np.arange(0, vert_num, dtype=np.uint32)
    # the vertex count for each vertex and the largest weight for each segment
    vert_cnt = np.ones(vert_num, dtype=np.uint32)
    # segment means
    seg_means = np.copy(verts)
    # segment covariances
    init_cov = np.eye(3) * 0.01 * 0.01  # 1 cm sigma
    seg_covs = np.tile(init_cov.reshape((1, 3, 3)), (vert_num, 1, 1))
    seg_normals = np.copy(normals)
    edge_sort_ids = np.argsort(edge_weights)
    for e_id in edge_sort_ids:
        w = edge_weights[e_id]
        if w > max_weight:
            continue
        v1_id, v2_id = edge_ids[e_id]
        s1_id = get_vertex_root(vert_tree, v1_id)
        s2_id = get_vertex_root(vert_tree, v2_id)
        if s1_id != s2_id:
            # try merge
            merged_mean, merged_cov = merge_mean_cov_est(
                seg_means[s1_id], seg_covs[s1_id], vert_cnt[s1_id],
                seg_means[s2_id], seg_covs[s2_id], vert_cnt[s2_id])
            merged_cnt = vert_cnt[s1_id] + vert_cnt[s2_id]
            w = (vert_cnt[s1_id] / merged_cnt)
            merged_normal = w * seg_normals[s1_id] + (1-w) * seg_normals[s2_id]
            if check_energy(merged_cov, seg_covs[s1_id], seg_covs[s2_id],
                            merged_cnt, vert_cnt[s1_id], vert_cnt[s2_id],
                            merged_normal, seg_normals[s1_id], seg_normals[s2_id],
                            max_num_vertices, weight_shape, weight_normal):
                vert_tree[s1_id] = s2_id
                vert_cnt[s2_id] = merged_cnt
                seg_means[s2_id] = merged_mean
                seg_covs[s2_id] = merged_cov
                seg_normals[s2_id] = merged_normal

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


def read_ply_scannet(ply_path):
    verts = None
    labels = None
    instances = None
    faces = None
    normals = None
    with open(ply_path, "rb") as f:
        plydata = PlyData.read(f)
        for el in plydata.elements:
            if el.name == "vertex":
                verts = np.stack([el.data["x"], el.data["y"], el.data["z"]], axis=1)
                if 'Nx' in el.data.dtype.names:
                    normals = np.stack([el.data["Nx"], el.data["Ny"], el.data["Nz"]], axis=1)
                is_label = False
                is_instance = False
                for pr in el.properties:
                    if pr._name == "label":
                        is_label = True
                    if pr._name == "instance":
                        is_instance = True
                if is_label:
                    labels = el.data["label"]
                if is_instance:
                    instances = el.data["instance"]
            if el.name == "face":
                flist = []
                for f in el.data:
                    flist.append(f[0])
                faces = np.asarray(flist)

    return verts, labels, instances, faces, normals, plydata


def read_ply_scenenn(ply_path):
    verts = None
    labels = None
    instances = None
    faces = None
    normals = None
    with open(ply_path, "rb") as f:
        plydata = PlyData.read(f)
        for el in plydata.elements:
            if el.name == "vertex":
                verts = np.stack([el.data["x"], el.data["y"], el.data["z"]], axis=1)
                if 'nx' in el.data.dtype.names:
                    normals = np.stack([el.data["nx"], el.data["ny"], el.data["nz"]], axis=1)
                is_label = False
                is_instance = False
                for pr in el.properties:
                    if pr._name == "nyu_class":
                        is_label = True
                    if pr._name == "instance":
                        is_instance = True
                if is_label:
                    labels = el.data["nyu_class"]
                if is_instance:
                    instances = el.data["instance"]
            if el.name == "face":
                flist = []
                for f in el.data:
                    flist.append(f[0])
                faces = np.asarray(flist)

    return verts, labels, instances, faces, normals, plydata


def segment_mesh(faces, normals, verts, params):
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
    # num_faces = faces.shape[0]
    # num_edges = 3 * num_faces
    # edge_weights = np.zeros(num_edges)
    edge_ids = np.concatenate([faces[:, 0:2], faces[:, 1:3], faces[:, [0, 2]]], axis=0)  # [3F, 2]

    # The following constants are required by the algorithm
    # The merge of C1 and C2 components with an edge of weight w between them
    # happens when w <= max_w (C1) + c / |C1|, and the same for C2.
    # In other words, the higher c, the bigger are the components
    # c = 0.1
    # c = params[0]  # k_segmentation
    # k = params[1]  # small_segment_size
    expected_num_vertices = params[1]
    max_weight = 1.0  # params[3]
    w_shape = 0.0  # params[4]
    w_normal = 0.0

    # After the main algorithms, one more step ensures that the small components
    # are merged to the bigger ones. So we merge C1 and C2 if
    # |c1| < k or |C2| < k.
    edge_weights = 1 - np.sum(
        normals[edge_ids[:, 0]] * normals[edge_ids[:, 1]], axis=1
    )  # very small?
    dp = verts[edge_ids[:, 1]] - verts[edge_ids[:, 0]]  # [3F, 3]
    conv_signs = (np.sum(dp * normals[edge_ids[:, 1]], axis=1) > 0)
    edge_weights[conv_signs] = edge_weights[conv_signs] * edge_weights[conv_signs]

    # vertex_tree, vertex_cnt, edge_sort_ids = segment_graph(
    # edge_ids, edge_weights, c, max_num_vertices, max_weight)

    vertex_tree, vertex_cnt, edge_sort_ids = segment_graph_energy(
        verts, normals, edge_ids, edge_weights, expected_num_vertices, max_weight, w_shape, w_normal)

    # edge_ids = edge_ids[edge_sort_ids]
    # edge_weights = edge_weights[edge_sort_ids]
    # merge
    # for e, w in zip(edge_ids, edge_weights):
    #     v1, v2 = e
    #     v1 = get_vertex_root(vertex_tree, v1)
    #     v2 = get_vertex_root(vertex_tree, v2)
    #     if v1 != v2 and (vertex_cnt[v1] < k or vertex_cnt[v2] < k):
    #         vertex_tree[v1] = v2
    #         vertex_cnt[v2] += vertex_cnt[v1]

    for i in range(0, len(vertex_tree)):
        get_vertex_root(vertex_tree, i)

    print("Segmentation took {} sec.".format(time.time() - t_start))
    return vertex_tree


def floodfill_mesh(faces, vertex_labels, is_respect_classes=False):
    n = len(vertex_labels)
    vertex_tree = np.arange(0, n)
    for f in faces:
        e1, e2, e3 = f
        for e in [[e1, e2], [e2, e3], [e3, e1]]:
            v1, v2 = e
            if (
                    vertex_labels[v2] < NUM_THINGS_CLASSES
                    and vertex_labels[v1] < NUM_THINGS_CLASSES
                    and (not is_respect_classes or vertex_labels[v2] == vertex_labels[v1])
            ):
                v1_root = get_vertex_root(vertex_tree, v1)
                v2_root = get_vertex_root(vertex_tree, v2)
                if v1_root != v2_root:
                    vertex_tree[v1_root] = v2_root
    for i in range(0, len(vertex_tree)):
        get_vertex_root(vertex_tree, i)
    return vertex_tree


def instances_floodfill(vertex_tree):
    root_mask = vertex_tree - vertex_tree[vertex_tree] == 0
    n = np.sum(root_mask)
    instances_root = np.arange(1, n + 1)
    instances = np.zeros_like(vertex_tree)
    instances[root_mask] = instances_root
    instances = instances[vertex_tree]
    return instances


def label_segments(segments, labels):
    seg_ids = np.unique(segments)
    for s_id in seg_ids:
        mask = segments == s_id
        class_ids, cnt = np.unique(labels[mask], return_counts=True)
        max_class_id = np.argmax(cnt)
        labels[mask] = class_ids[max_class_id]


def instances_from_semantics_floodfill(
        faces, labels, plydata, out_path, is_respect_classes=False
):
    vertex_tree = floodfill_mesh(faces, labels, is_respect_classes=is_respect_classes)
    instances = instances_floodfill(vertex_tree)
    instances[labels >= NUM_THINGS_CLASSES] = 0
    # replace labels in the mesh, add instances to the mesh
    for el in plydata.elements:
        if el.name == "vertex":
            el.data["label"] = labels

    v = plydata.elements[0]
    f = plydata.elements[1]

    # Create the new vertex data with appropriate dtype
    a = np.empty(len(v.data), v.data.dtype.descr + [("instance", "i4")])
    for name in v.data.dtype.fields:
        a[name] = v[name]

    # Recreate the PlyElement instance
    v = PlyElement.describe(a, "vertex")
    v.data["instance"] = instances.astype(np.uint32)

    # Recreate the PlyData instance
    plydata = PlyData([v, f], text=True)
    plydata.write(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Add instance information and refine semantic labels"
    parser.add_argument(
        "-i",
        "--in",
        required=True,
        help=("Path to the input point cloud with semantics"),
    )

    parser.add_argument(
        "-o",
        "--out",
        required=True,
        help=("Path to the output point cloud with refined semantics and instances"),
    )

    parser_args = vars(parser.parse_args())
    input_path = parser_args["in"]
    output_path = parser_args["out"]

    verts, labels, instances, faces, normals, plydata = read_ply_scennet(input_path)

    vertex_tree = segment_mesh(faces, normals)
    segments = instances_floodfill(vertex_tree)

    labels_ref = np.copy(labels)
    label_segments(segments, labels_ref)

    instances_from_semantics_floodfill(
        faces, labels_ref, plydata, output_path, is_respect_classes=True
    )