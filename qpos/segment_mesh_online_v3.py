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
# from qpos.v2.segment_mesh import reproject_segments, compute_normals, 
    # colour_map_using_hash, colourize_instances, estimate_normals_from_depth, colour_map_using_hash_ids
# from qpos.v2.geometric_instances import read_ply_scannet, get_vertex_root
from qpos.geometric_instances import try_merge_using_vertex_count, get_vertex_root
from qpos.segment_mesh import reproject_segments, compute_normals_o3d, colour_map_using_hash, colourize_instances, merge_small_segments
from dataio.utils import read_ply
from copy import deepcopy
import time
import open3d as o3d
import numpy as np
import torch
import os
import sys
sys.path.append("..")


"""
TODO: This script is actually using v2...
"""


def read_camera_pose(file_path):
    camera_pose = np.loadtxt(file_path)
    return camera_pose


def createOpen3DIntrinsic(K, w, h):
    """
    Create the Open3D Intrinsics object
    :param K the intrinsics matrix
    :param w width
    :param h height
    :returns intrinsic the intrisic object
    """
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = K
    intrinsic.width = w
    intrinsic.height = h
    return intrinsic


def convert_to_open3d_param(K, T, w, h):
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = createOpen3DIntrinsic(K, w, h)
    param.extrinsic = T
    return param


def convert_to_open3d_param_arbitrary(intrinsic, extrinsic):
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
    param.intrinsic.intrinsic_matrix = intrinsic
    param.extrinsic = extrinsic
    return param


def save_vertex_mask(plydata, mask, current_output_path):
    for el in plydata.elements:
        if el.name == "vertex":
            el.data["red"][mask] = 255
            el.data["red"][~mask] = 0
            el.data["green"] = 0
            el.data["blue"] = 0
    plydata.write(current_output_path)


def incremental_frame_segmentation(verts, normals, faces, verts_mask, segdata, max_num_vertices, small_size):
    is_reprojection = True

    unlabelled_vertex_mask = segdata.vert_cnt[segdata.segments] == 1
    unlbl_vis_vertex_mask = unlabelled_vertex_mask & verts_mask

    faces_mask = unlbl_vis_vertex_mask[faces[:, 0]
                                       ] | unlbl_vis_vertex_mask[faces[:, 1]] | unlbl_vis_vertex_mask[faces[:, 2]]
    faces_current = faces[faces_mask]
    num_faces = faces_current.shape[0]
    num_edges = 3 * num_faces
    edge_weights = np.zeros(num_edges)
    edge_ids = np.concatenate(
        [faces_current[:, 0:2], faces_current[:, 1:3], faces_current[:, [0, 2]]], axis=0)
    unlbl_vis_vertex_mask[edge_ids[:, 0]] = True
    unlbl_vis_vertex_mask[edge_ids[:, 1]] = True
    
    max_weight = 1.0
    w_shape = 0.0
    w_normal = 0.0

    edge_weights = 1 - np.sum(
        normals[edge_ids[:, 0]] * normals[edge_ids[:, 1]], axis=1
    )
    dp = verts[edge_ids[:, 1]] - verts[edge_ids[:, 0]]
    conv_signs = (np.sum(dp * normals[edge_ids[:, 1]], axis=1) > 0)
    edge_weights[conv_signs] = edge_weights[conv_signs] * \
        edge_weights[conv_signs]

    t_before_sort = time.time()
    edge_sort_ids = np.argsort(edge_weights)
    t_after_sort = time.time()
    # print('sort took {}'.format(t_after_sort-t_before_sort))
    for e_id in edge_sort_ids:
        w = edge_weights[e_id]
        if w > max_weight:
            continue
        v1_id, v2_id = edge_ids[e_id]
        s1_id = segdata.segments[v1_id]
        s2_id = segdata.segments[v2_id]
        s1_id = get_vertex_root(segdata.seg_tree, s1_id)
        s2_id = get_vertex_root(segdata.seg_tree, s2_id)
        if s1_id != s2_id:
            # try merge
            try_merge_using_vertex_count(segdata.seg_tree, segdata.vert_cnt, segdata.seg_means,
                                         segdata.seg_covs, segdata.seg_normals,
                                         max_num_vertices, s1_id, s2_id)
            # try merge
            # merged_mean, merged_cov = merge_mean_cov_est(
            #     segdata.seg_means[s1_id], segdata.seg_covs[s1_id], segdata.vert_cnt[s1_id],
            #     segdata.seg_means[s2_id], segdata.seg_covs[s2_id], segdata.vert_cnt[s2_id])
            # merged_cnt = segdata.vert_cnt[s1_id] + segdata.vert_cnt[s2_id]
            # w = (segdata.vert_cnt[s1_id] / merged_cnt)
            # merged_normal = w * \
            #     segdata.seg_normals[s1_id] + (1-w) * segdata.seg_normals[s2_id]
            # if check_energy(merged_cov, segdata.seg_covs[s1_id], segdata.seg_covs[s2_id],
            #                 merged_cnt, segdata.vert_cnt[s1_id], segdata.vert_cnt[s2_id],
            #                 merged_normal, segdata.seg_normals[s1_id], segdata.seg_normals[s2_id],
            #                 expected_num_vertices, w_shape, w_normal):
            #     segdata.seg_tree[s1_id] = s2_id
            #     segdata.vert_cnt[s2_id] = merged_cnt
            #     segdata.seg_means[s2_id] = merged_mean
            #     segdata.seg_covs[s2_id] = merged_cov
            #     segdata.seg_normals[s2_id] = merged_normal
            #     segdata.segments[v1_id] = s2_id
            #     segdata.segments[v2_id] = s2_id
            
    t_after_loop = time.time()
    print('loop took {}'.format(t_after_loop-t_after_sort))

    t_before_ff = time.time()
    seg_ids = np.arange(0, len(segdata.segments))
    active_seg_mask = (
        segdata.seg_tree[segdata.seg_tree] - segdata.seg_tree != 0)
    for s_id in seg_ids[active_seg_mask]:
        segdata.seg_tree[s_id] = get_vertex_root(segdata.seg_tree, s_id)
    segdata.segments = segdata.seg_tree[segdata.segments]
    t_after_ff = time.time()
    print('ff took {}'.format(t_after_ff-t_before_ff))

    if is_reprojection:
        # segment_v1 statistics (mean vertex, normal, covariances)
        # active_seg_mask = (segdata.segments > 0)

        # seg_centers, seg_normals, seg_covs, normal_covs = \
        #     get_segment_statistics(segdata.segments[active_seg_mask], verts[active_seg_mask],
        #     normals[active_seg_mask], min_seg_size=small_segment_size)

        segment_ids = np.unique(segdata.segments[unlbl_vis_vertex_mask])
        seg_mask = np.zeros(len(segdata.segments), dtype=np.bool)
        seg_mask[segment_ids] = True
        ext_vert_mask = seg_mask[segdata.segments]
        assert (
            np.sum(segdata.seg_tree[segment_ids] - segment_ids == 0) == len(segment_ids))
        if (len(segment_ids) > 1) and (np.sum(ext_vert_mask) > 0):
            t_after_stats = time.time()
            verts_masked = verts[ext_vert_mask]
            normals_masked = normals[ext_vert_mask]
            seg_ids_per_vertex = segdata.segments[ext_vert_mask]
            # segments_masked = reproject_segments(verts_masked, normals_masked, segdata.seg_means,
                                                #  segdata.seg_covs, segdata.seg_normals, segment_ids)
            segments_masked = reproject_segments(verts_masked, normals_masked, segdata.seg_means,
                                                 segdata.seg_covs, segdata.seg_normals, seg_ids=seg_ids_per_vertex)
            segdata.segments[ext_vert_mask] = segments_masked
            t_after_reproj = time.time()
            print('reproject took {}'.format(t_after_reproj-t_after_stats))

        merge_small_segments(edge_ids, segdata.segments, segdata.vert_cnt, small_size, seg_means=segdata.seg_means, seg_covs=segdata.seg_covs, seg_normals=segdata.seg_normals, segment_tree=segdata.seg_tree)


def depth_segmentation(depth, T_WC, K4, segdata):
    depth_mask = (depth > 0)
    z = depth.reshape(-1, 1)
    h, w = depth.shape[0:2]
    u = np.tile(np.arange(0, w).reshape(1, -1), (h, 1)).reshape(-1, 1)
    v = np.tile(np.arange(0, h).reshape(-1, 1), (1, w)).reshape(-1, 1)
    points_h = np.concatenate([u*z, v*z, z, np.ones((len(z), 1))], axis=1)
    points_mask = depth_mask.reshape(-1)
    P = T_WC @ np.linalg.inv(K4)
    points_world_h = points_h @ P.transpose()
    points_world = points_world_h[:, 0:3]
    normals_world, normals_mask = estimate_normals_from_depth(
        depth, points_world, points_mask)

    normals_vis = ((normals_world.reshape((h, w, 3)) + 1)
                   * 255.0/2.0).astype(np.uint8)
    normals_vis_flat = normals_vis.reshape(-1, 3)
    normals_vis_flat[~normals_mask] = 0
    # print('saved')
    # cv2.imwrite('/data/tmp_out/normals_debug/normals_mask.png', (normals_mask.reshape(h, w) * 255).astype(np.uint8))
    # cv2.imwrite('/data/tmp_out/normals_debug/normals.png', normals_vis)

    points_mask = points_mask & normals_mask.reshape(-1)
    t_start_reproj = time.time()
    segments_masked = reproject_segments(points_world[points_mask], normals_world[points_mask],
                                         segdata.seg_means, segdata.seg_covs, segdata.seg_normals)
    t_end_reproj = time.time()
    print('depth segmentation took ' + str(t_end_reproj - t_start_reproj))
    segments = np.ones((len(z)), dtype=np.int32) * (-1)
    segments[points_mask] = segments_masked
    return segments.reshape((h, w))


class IncrementalSegmenter:
    def __init__(self, verts, faces, normals, colors, trajectory, segmentation_params,
                 K, segmentation_processor, path_to_depth, skip=10, depth_diff_threshold=0.1, max_num_vertices=240,
                 small_segment_size=120):
        """
        :param verts: full mesh vertices
        :param faces:
        :param normals:
        :param colors:
        :param trajectory: list of c2w
        :param segmentation_params: segment_size, etc.
        :param K: camera intrinsics
        :param segmentation_processor: segmentation logger
        :param path_to_depth:
        :param skip: simulating mapping_every
        """
        self.current_frame_id = 0
        vert_num = verts.shape[0]
        self.seg_tree = np.arange(0, vert_num, dtype=np.uint32)
        self.segments = np.copy(self.seg_tree)
        self.obs_mask = np.zeros(vert_num, dtype=np.bool)
        self.seg_means = np.copy(verts)
        self.seg_covs = np.zeros((vert_num, 3, 3))
        self.vert_cnt = np.ones(vert_num, dtype=np.uint32)
        self.seg_normals = np.copy(normals)
        self.segmentation_params = segmentation_params
        self.max_num_vertices = max_num_vertices
        self.small_segment_size = small_segment_size
        self.K4 = np.eye(4)
        self.K4[0:3, 0:3] = K
        self.mesh_vertices = verts
        self.mesh_faces = faces
        self.mesh_normals = normals
        self.mesh_colors = colors
        self.trajectory = trajectory
        self.sequence_length = len(self.trajectory)
        self.segmentation_processor = segmentation_processor
        self.path_to_depth = path_to_depth
        self.skip = skip
        self.depth_diff_threshold = depth_diff_threshold

    def check_pose_valid(self, pose):
        if np.isnan(pose).any().item() or np.isinf(pose).any():
            # print("Skipping frame: {}".format(frame_id))
            return False
        else:
            return True

    def move_forward(self, vis):
        mf_start = time.time()
        print("Capture image {:06d}".format(self.current_frame_id * self.skip))
        depth_rendered = np.asarray(
            vis.capture_depth_float_buffer(do_render=True))
        n_vertices = self.mesh_vertices.shape[0]
        # [V, 4]
        mesh_vertices_h = np.concatenate(
            [self.mesh_vertices, np.ones((n_vertices, 1))], axis=1)
        T = self.trajectory[self.current_frame_id]
        if self.check_pose_valid(T):
            KT = self.K4 @ T
            # [V, 4]
            mesh_vertices_cam_h = mesh_vertices_h @ KT.transpose()
            uv = mesh_vertices_cam_h[:, 0:2] / \
                mesh_vertices_cam_h[:, 2].reshape(-1, 1)
            u = uv[:, 0]
            v = uv[:, 1]
            u = np.clip(u, 0, depth_rendered.shape[1] - 1).astype(np.int32)
            v = np.clip(v, 0, depth_rendered.shape[0] - 1).astype(np.int32)
            z = mesh_vertices_cam_h[:, 2]
            z_sampled = depth_rendered[v, u]
            # z_diff = np.abs(z - z_sampled)
            # mask_vertices = (z_diff < depth_diff_threshold)
            mask_vertices = ((z > 0) & (
                z < (z_sampled + self.depth_diff_threshold)))
            mask_vertices = mask_vertices & ~((u <= 0) | (v <= 0) | (
                u >= depth_rendered.shape[1] - 1) | (v >= depth_rendered.shape[0] - 1))
            is_start = time.time()
            print(f'Segmenting max {self.max_num_vertices} small {self.small_segment_size}')
            incremental_frame_segmentation(self.mesh_vertices, self.mesh_normals, self.mesh_faces,
                                           mask_vertices, self, self.max_num_vertices, small_size=self.small_segment_size)
            is_finish = time.time()
            print(f'inc seg took {is_finish-is_start} s')
            # depth_file_path = self.path_to_depth + '/frame-{:06d}.depth.pgm'.format(self.current_frame_id)
            # depth_loaded = cv2.imread(depth_file_path, -1).astype(np.float32) * 0.001

            # depth_segmentation(depth_loaded, T, self.K4, self)
        else:
            mask_vertices = None

        segments_flt = np.copy(self.segments).astype(np.int32)
        invalid_seg_mask = (self.vert_cnt[self.segments] == 1)
        segments_flt[invalid_seg_mask] = -1
        is_last_frame = (self.current_frame_id == self.sequence_length - 1)

        # self.segmentation_processor.process(segments_flt, self.current_frame_id, is_last_frame)
        if mask_vertices is not None:
            self.obs_mask |= mask_vertices
        self.segmentation_processor.process(self, segments_flt, self.skip * self.current_frame_id,
                                            is_last_frame, inside_current_view=mask_vertices, mask_partial=self.obs_mask.astype(np.int32))

        if self.current_frame_id < self.sequence_length - 1:
            if self.current_frame_id < self.sequence_length - 1:  # self.skip:
                self.current_frame_id += 1  # self.skip
            else:
                self.current_frame_id = self.sequence_length - 1
            # height, width = depth_rendered.shape[0:2]
            vc = vis.get_view_control()
            # vc.convert_from_pinhole_camera_parameters(
            #     convert_to_open3d_param(self.K4[0:3, 0:3],
            #                             self.trajectory[self.current_frame_id], width, height))
            vc.convert_from_pinhole_camera_parameters(
                convert_to_open3d_param_arbitrary(
                    self.K4[0:3, 0:3], self.trajectory[self.current_frame_id]),
                allow_arbitrary=True)
        else:
            vis.destroy_window()
        mf_finish = time.time()
        # print(f'move forward took {mf_finish-mf_start} s')
        return False


def process_sequence_with_segmenter(scene_dir, segmentation_processor, path_to_mesh, path_to_depth, path_to_intri, max_num_vertices=240,
                                    small_segment_size=120, width=640, height=480, skip=10, depth_diff_threshold=0.1):
    # verts, labels, instances, faces, normals, plydata = read_ply_scannet(
        # path_to_mesh)
    mesh_data = read_ply(path_to_mesh)

    verts, labels, faces, normals, plydata = mesh_data["verts"], mesh_data["labels"], mesh_data["faces"], mesh_data["normals"], mesh_data["plydata"]

    if normals is None:
        print("Using open3d normal computation!")
        normals = compute_normals_o3d(verts, faces)
    colors = np.stack([np.asarray(plydata.elements[0]["red"]),
                       np.asarray(plydata.elements[0]["green"]),
                       np.asarray(plydata.elements[0]["blue"])], axis=1).astype(np.float32) / 255.

    K = np.loadtxt(path_to_intri)[:3, :3].astype(np.float32)
    scene_pose_dir = os.path.join(scene_dir, "pose")
    sequence_length = len(os.listdir(scene_pose_dir))
    trajectory = np.zeros((sequence_length, 4, 4))
    for frame_id in range(0, sequence_length):
        T_CW = read_camera_pose(os.path.join(
            scene_pose_dir, "{}.txt".format(frame_id * skip)))
        trajectory[frame_id] = np.linalg.inv(T_CW)

    # v2 parameters
    segmentation_params = None

    # K[0, 2] = width / 2 - 0.5
    # K[1, 2] = height / 2 - 0.5
    inc_segmenter = IncrementalSegmenter(verts, faces, normals, colors, trajectory, segmentation_params,
                                         K, segmentation_processor, path_to_depth, skip=skip,
                                         depth_diff_threshold=depth_diff_threshold, max_num_vertices=max_num_vertices,
                                         small_segment_size=small_segment_size)

    # read mesh
    mesh = o3d.io.read_triangle_mesh(path_to_mesh)
    # visualize mesh for each frame
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="myname", visible=False,
                      width=width, height=height)
    ro = vis.get_render_option()
    vis.add_geometry(mesh)
    vc = vis.get_view_control()
    # camera_new = convert_to_open3d_param(K, trajectory[0], width, height)
    camera_new = convert_to_open3d_param_arbitrary(K, trajectory[0])
    vc.convert_from_pinhole_camera_parameters(camera_new, allow_arbitrary=True)

    vis.register_animation_callback(inc_segmenter.move_forward)
    vis.run()
    print('yahoo')


class SegmentationLogger:
    def __init__(self, debug_output_path, path_to_mesh, mapping_every=10):
        self.debug_output_path = debug_output_path
        mesh_dict = read_ply(
            path_to_mesh)
        self.plydata = mesh_dict['plydata']
        self.plydata_mask = deepcopy(self.plydata)
        self.mapping_every = mapping_every
        self.plydata.text = False
        self.plydata.byte_order = "<"
        self.colour_map_seg = colour_map_using_hash(100)
        self.max_seg_id = 100
        self.is_save_all = True

    def process(self, segdata, segments_partial, current_frame_id, is_last_frame, inside_current_view=None, mask_partial=None):
        """
        :param segments_partial: global segment ids [V_global,]
        :param current_frame_id:
        :param is_last_frame:
        :param mask_partial: observed mask [V_global,]
        :return:
        """
        if (current_frame_id % self.mapping_every == 0) or is_last_frame:
            t_save_start = time.time()
            save_current_frame_dir = os.path.join(
                self.debug_output_path, "{:06d}".format(current_frame_id))
            if not os.path.exists(save_current_frame_dir):
                os.makedirs(save_current_frame_dir)
            if self.is_save_all:
                seg_ids = np.unique(segments_partial)
                if np.max(seg_ids) > self.max_seg_id:
                    self.colour_map_seg = colour_map_using_hash(np.max(seg_ids)+1)
                t_end_cmap = time.time()
                # print(f'cmap took {t_end_cmap-t_save_start} s')
                colourize_instances(self.plydata, self.colour_map_seg,
                                    segments_partial, min_value=0)

            # save segments.pth already contains segmented_mask
            torch.save(torch.from_numpy(segments_partial), os.path.join(
                save_current_frame_dir, "segments.pth"))
            # save current segment mesh (visualisation only)
            if self.is_save_all:
                self.plydata.write(os.path.join(
                    save_current_frame_dir, "{:06d}_segments.ply".format(current_frame_id)))
                t_save_seg_end = time.time()
                # print(f'save seg took {t_save_seg_end-t_save_start}')

                # save observation mask for visualisation
            if mask_partial is not None:
                torch.save(torch.from_numpy(mask_partial.astype(bool)),
                           os.path.join(save_current_frame_dir, "obs_mask.pth"))
                if self.is_save_all:
                    obs_id = np.max(mask_partial)
                    colour_map_mask = colour_map_using_hash(obs_id)
                    colourize_instances(self.plydata_mask,
                                        colour_map_mask, mask_partial, min_value=1)
                    self.plydata_mask.write(os.path.join(
                        save_current_frame_dir, "{:06d}_mask.ply".format(current_frame_id)))

                    t_save_mask_end = time.time()
                    # print(f'save mask took {t_save_mask_end-t_save_seg_end}')
                if inside_current_view is not None:
                    torch.save(torch.from_numpy(inside_current_view), os.path.join(
                        save_current_frame_dir, "frustum_mask.pth"))
                t_save_end = time.time()
                # print(f'save took {t_save_end-t_save_start}')


def run():
    """
    Minimal requirement for sequential mapping:
    1. segment.pth: a tensor of [V,] saving per-vertex segment_id
    2. segment.ply (optional): mesh visualizing the segments
    3. valid_segment_mask.pth: a tensor of [N_seg_all,] saving mask for valid segment ids.
    We define invalid segments as segments that contain too few vertices, e.g. <10
    4. knn_mat.pth: a tensor of [N_seg_valid, K] saving the KNN-relationship for every valid segment
    For sequential case, we also need to propagate all those things instead of doing everything from scratch.
    """
    from config import get_scannet_root
    scannet_root = get_scannet_root()
    scene = "scene0217_00"
    scene_dir = os.path.join(scannet_root, scene)
    mapping_every = 20
    out_root = "../logs/experiments/20230528_sequential_mapping/scannet"
    output_path = os.path.join(
        out_root, "{}_skip{}".format(scene, mapping_every))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    path_to_depth = os.path.join(scene_dir, "depth")
    path_to_mesh = os.path.join(scene_dir, "{}_vh_clean_2.ply".format(scene))

    # segmentation_logger is an example of how to get incremental segmentations.
    # Now it is simply saving them into a ply file for the illustration purposes.
    segmentation_logger = SegmentationLogger(
        output_path, path_to_mesh, mapping_every=mapping_every)
    process_sequence_with_segmenter(
        scene_dir,
        segmentation_logger,
        path_to_mesh,
        path_to_depth,
        width=640,
        height=480,
        skip=mapping_every)


if __name__ == '__main__':
    run()
