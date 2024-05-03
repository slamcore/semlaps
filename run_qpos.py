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
import ray
import time
import numpy as np
from tqdm import tqdm
from qpos.segment_mesh import segment_one_scene
from dataio.utils import get_scene_list


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def segment_meshes(hyper_qp, dataset_type, dataset_root, scene_list, mesh_name_suffix):
    errors, avg_seg_sizes, min_seg_sizes, max_seg_sizes, n_small_segs, n_segs, avg_d_spats = dict(), dict(), dict(), dict(), dict(), dict(), dict()
    for scene in tqdm(scene_list):
        scene_dir = os.path.join(dataset_root, scene)
        mesh_name = scene + mesh_name_suffix
        err, avg_seg_size, min_seg_size, max_seg_size, n_small_seg, n_seg, avg_d_spat = segment_one_scene(dataset_type, scene_dir, mesh_name, **hyper_qp)
        errors[scene] = err
        avg_seg_sizes[scene] = avg_seg_size
        min_seg_sizes[scene] = min_seg_size
        max_seg_sizes[scene] = max_seg_size
        n_small_segs[scene] = n_small_seg
        n_segs[scene] = n_seg
        avg_d_spats[scene] = avg_d_spat

    return errors, avg_seg_sizes, min_seg_sizes, max_seg_sizes, n_small_segs, n_segs, avg_d_spats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_suffix", type=str, required=True, help="sub_dir to save the segments")
    parser.add_argument("--is_reprojection", type=bool, default=True)
    parser.add_argument("--small_segment_size", type=float, default=30)
    parser.add_argument("--expected_segment_size", type=float, default=60)
    parser.add_argument("--dataset_type", type=str, default="scannet")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--n_proc", type=int, default=12)
    args = parser.parse_args()

    # Hyper-parameters for quasi-planar segment
    hyper_qp = {
        "segment_suffix": args.segment_suffix,
        "is_reprojection": args.is_reprojection,
        "small_segment_size": args.small_segment_size,
        "expected_segment_size": args.expected_segment_size,
        "is_v3": True,
        "o3d_normal": True,
        "debug": False
    }

    start = time.time()
    n_proc = args.n_proc
    segment_meshes_remote = ray.remote(segment_meshes)
    ray.init()
    
    dataset_type = args.dataset_type
    dataset_root = args.dataset_root
    if dataset_type == "scannet":
        scene_split_train = "configs/scannetv2_train.txt"
        scene_split_val = "configs/scannetv2_val.txt"
        scene_list = get_scene_list(scene_split_train) + get_scene_list(scene_split_val)
        mesh_name_suffix = "_vh_clean_2.labels.ply"
    elif dataset_type == "scannet_test":
        scene_split_test = "configs/scannetv2_test.txt"
        scene_list = get_scene_list(scene_split_test)
        mesh_name_suffix = "_vh_clean_2.ply"
    elif dataset_type == "slamcore":
        scene_list = get_scene_list("configs/slamcore.txt")
        mesh_name_suffix = "_clean.labels.ply"
        n_proc = 4
    else:
        raise NotImplementedError

    scene_lists = split_list(scene_list, n_proc)
    futures = [segment_meshes_remote.remote(hyper_qp, dataset_type, dataset_root, scene_lists[w_id], mesh_name_suffix) for w_id in range(n_proc)]

    # gather results
    results = ray.get(futures)
    errors, avg_seg_sizes, min_seg_sizes, max_seg_sizes, n_small_segs, n_segs, avg_d_spats = dict(), dict(), dict(), dict(), dict(), dict(), dict()
    for rst in results:
        errors = errors | rst[0]
        avg_seg_sizes = avg_seg_sizes | rst[1]
        min_seg_sizes = min_seg_sizes | rst[2]
        max_seg_sizes = max_seg_sizes | rst[3]
        n_small_segs = n_small_segs | rst[4]
        n_segs = n_segs | rst[5]
        avg_d_spats = avg_d_spats | rst[6]


    filename_params = "{}_{}".format(hyper_qp["small_segment_size"], hyper_qp["expected_segment_size"])

    result_filename = "{}_segment_{}.txt".format(dataset_type, filename_params)
    with open(os.path.join(dataset_root, result_filename), "w") as f:
        f.write("-------------Evaluation Result--------------\n")
        for scene in avg_seg_sizes:
            f.write("{}: {:.2f}%, {:.4f}, {}, {}, ({}, {}, {}, {}), {}, {:.4f}\n".format(scene,
                                                                                         errors[scene],
                                                                                         avg_seg_sizes[scene],
                                                                                         min_seg_sizes[scene],
                                                                                         max_seg_sizes[scene],
                                                                                         n_small_segs[scene][0],
                                                                                         n_small_segs[scene][1],
                                                                                         n_small_segs[scene][2],
                                                                                         n_small_segs[scene][3],
                                                                                         n_segs[scene],
                                                                                         avg_d_spats[scene]))

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
        f.write("Average segment_v2 size stats:\n")
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

        spat_list = np.array(list(avg_d_spats.values()))
        spat_mean = spat_list.mean()
        spat_std = spat_list.std()
        spat_max = spat_list.max()
        spat_min = spat_list.min()
        f.write("Average segment spat stats:\n")
        f.write("mean: {:.4f}\n".format(spat_mean))
        f.write("std: {:.4f}\n".format(spat_std))
        f.write("max: {:.4f}\n".format(spat_max))
        f.write("min: {:.4f}\n".format(spat_min))

        end = time.time()

        print("Total time elapsed: {} seconds".format(end - start))
