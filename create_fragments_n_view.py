
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
import imageio
import numpy as np
from tqdm import tqdm

from dataio.utils import nyu40_to_scannet20
from config import get_scannet_root


"""
This is the script used to create multi-view frames for scannet scenes
"""


def create_fragment_for_one_scene_batched(scannet_dir, scene, save_dir, n_views, max_move=80, skip=20, step=1, filter=True):
    """
    create sub-sequences for batched inference, i.e. take the middle view as reference view and put it to the first
    :param scannet_dir:
    :param scene:
    :param save_dir:
    :param n_views:
    :param max_move:
    :param skip:
    :param step:
    :param filter:
    :return:
    """
    label_dir = os.path.join(scannet_dir, scene, "label-240")
    pose_dir = os.path.join(scannet_dir, scene, "pose")
    n_frames = len(os.listdir(pose_dir))

    # step 1: get valid frames
    valid_frame_ids = []
    for frame_id in range(0, n_frames, skip):
        c2w = np.loadtxt(os.path.join(pose_dir, "{}.txt".format(frame_id)))
        if np.isnan(c2w).any() or np.isinf(c2w).any():
            continue

        if skip == 20 and filter:
            if not os.path.exists(os.path.join(label_dir, "{}.png".format(frame_id))):
                continue
            label = np.array(imageio.imread(os.path.join(label_dir, "{}.png".format(frame_id))))
            h, w = label.shape
            N = h * w
            label = nyu40_to_scannet20(label)
            num_labelled_1 = np.count_nonzero(label > 0)
            label[label > 20] = 0
            num_labelled_2 = np.count_nonzero(label > 0)
            # print(num_labelled_1, num_labelled_2, num_labelled_2 / N)
            if (num_labelled_2 / N) < 0.1:
                continue

        valid_frame_ids.append(frame_id)

    # [N,]
    valid_frame_ids = np.asarray(valid_frame_ids)

    fragments = []
    # ref_view must cover as many images as possible
    for ref_view in valid_frame_ids[::step]:
        # let i be ref-view and find n nearest views
        dist = np.abs(valid_frame_ids - ref_view)
        nn_views_id = np.argsort(dist)[1:n_views]
        candidates = [ref_view]
        for nn_id in nn_views_id:
            if dist[nn_id] <= max_move:
                candidates.append(valid_frame_ids[nn_id])

        if len(candidates) == n_views:
            fragments.append(candidates)
        else:
            n_sample = n_views - len(candidates)
            ids = list(np.random.randint(len(candidates), size=n_sample))
            sampled_views = [candidates[i] for i in ids]
            fragments.append(candidates + sampled_views)

    with open(os.path.join(save_dir, "{}.txt".format(scene)), "w") as f:
        for fragment in fragments:
            things_to_write = ""
            for i, view in enumerate(fragment):
                things_to_write += "{}".format(view)
                if i < n_views - 1:
                    things_to_write += " "
                else:
                    things_to_write += "\n"
            f.write(things_to_write)


def create_fragments_batched(scannet_root, save_files_root, filter=False, n_views=3, step=1, skip=20):
    # TODO: hard-coded
    scene_files = ["configs/scannetv2_train.txt", "configs/scannetv2_val.txt"]
    suffix = "filtered" if filter else "all"
    save_files_dir = os.path.join(save_files_root, "skip_{}/{}_views_step_{}/{}".format(skip, n_views, step, suffix))
    os.makedirs(save_files_dir, exist_ok=True)

    all_scenes_list = []
    for scene_file in scene_files:
        with open(scene_file, "r") as f:
            scenes = f.readlines()
            for scene in scenes:
                scene = scene.strip()  # remove \n
                all_scenes_list.append(scene)

    for scene in tqdm(all_scenes_list):
        print("Processing {}".format(scene))
        create_fragment_for_one_scene_batched(scannet_root, scene, save_files_dir, n_views, max_move=80, skip=skip, step=step, filter=filter)


def create_fragment_for_one_scene_causal(scannet_dir, scene, save_dir, n_views, max_move=80, skip=20, step=1, filter=True):
    """
    create sub-sequences for causal inference, i.e. take the last view as reference view and put it to the first
    :param scannet_dir:
    :param scene:
    :param save_dir:
    :param n_views:
    :param max_move:
    :param skip:
    :param step:
    :param filter:
    :return:
    """
    label_dir = os.path.join(scannet_dir, scene, "label-240")
    pose_dir = os.path.join(scannet_dir, scene, "pose")
    n_frames = len(os.listdir(pose_dir))

    # step 1: get valid frames
    valid_frame_ids = []
    for frame_id in range(0, n_frames, skip):
        c2w = np.loadtxt(os.path.join(pose_dir, "{}.txt".format(frame_id)))
        if np.isnan(c2w).any() or np.isinf(c2w).any():
            continue

        if skip == 20 and filter:
            if not os.path.exists(os.path.join(label_dir, "{}.png".format(frame_id))):
                continue
            label = np.array(imageio.imread(os.path.join(label_dir, "{}.png".format(frame_id))))
            h, w = label.shape
            N = h * w
            label = nyu40_to_scannet20(label)
            num_labelled_1 = np.count_nonzero(label > 0)
            label[label > 20] = 0
            num_labelled_2 = np.count_nonzero(label > 0)
            # print(num_labelled_1, num_labelled_2, num_labelled_2 / N)
            if (num_labelled_2 / N) < 0.1:
                continue

        valid_frame_ids.append(frame_id)

    # [N,]
    valid_frame_ids = np.asarray(valid_frame_ids)

    fragments = []
    for ref_view in valid_frame_ids[::step]:
        # let i be ref-view and find n nearest views
        dist = valid_frame_ids - ref_view
        # set frame_ids after ref_id to have very large dist, s.t. smaller ids always appear in first n views
        dist[dist > 0] = 1000000
        dist = np.abs(dist)
        nn_views_id = np.argsort(dist)[1:n_views]
        candidates = [ref_view]
        for nn_id in nn_views_id:
            if dist[nn_id] <= max_move:
                candidates.append(valid_frame_ids[nn_id])

        if len(candidates) == n_views:
            fragments.append(candidates)
        else:
            n_sample = n_views - len(candidates)
            if len(candidates) == 1:
                sampled_views = [ref_view] * n_sample
            else:
                # shouldn't re-sample the reference view
                ids = list(np.random.randint(1, len(candidates), size=n_sample))
                sampled_views = [candidates[i] for i in ids]
            fragments.append(candidates + sampled_views)

    with open(os.path.join(save_dir, "{}.txt".format(scene)), "w") as f:
        for fragment in fragments:
            things_to_write = ""
            for i, view in enumerate(fragment):
                things_to_write += "{}".format(view)
                if i < n_views - 1:
                    things_to_write += " "
                else:
                    things_to_write += "\n"
            f.write(things_to_write)


def create_fragments_causal(scannet_root, save_files_root, filter=False, n_views=3, step=1, skip=20):
    scene_files = ["configs/scannetv2_train.txt", "configs/scannetv2_val.txt"]
    suffix = "causal"
    save_files_dir = os.path.join(save_files_root, "skip_{}/{}_views_step_{}/{}".format(skip, n_views, step, suffix))
    os.makedirs(save_files_dir, exist_ok=True)

    all_scenes_list = []
    for scene_file in scene_files:
        with open(scene_file, "r") as f:
            scenes = f.readlines()
            for scene in scenes:
                scene = scene.strip()  # remove \n
                all_scenes_list.append(scene)

    for scene in tqdm(all_scenes_list):
        print("Processing {}".format(scene))
        create_fragment_for_one_scene_causal(scannet_root, scene, save_files_dir, n_views, max_move=80, skip=skip, step=step, filter=filter)


def create_valid_frame_list(scannet_dir, scene, save_dir, skip=20, filter=True):
    """
    create sub-sequences for batched inference, i.e. take the middle view as reference view and put it to the first
    :param scannet_dir:
    :param scene:
    :param save_dir:
    :param skip:
    :param filter:
    :return:
    """
    label_dir = os.path.join(scannet_dir, scene, "label-240")
    pose_dir = os.path.join(scannet_dir, scene, "pose")
    n_frames = len(os.listdir(pose_dir))

    # step 1: get valid frames
    valid_frame_ids = []
    for frame_id in range(0, n_frames, skip):
        c2w = np.loadtxt(os.path.join(pose_dir, "{}.txt".format(frame_id)))
        if np.isnan(c2w).any() or np.isinf(c2w).any():
            continue

        if skip == 20 and filter:
            if not os.path.exists(os.path.join(label_dir, "{}.png".format(frame_id))):
                continue
            label = np.array(imageio.imread(os.path.join(label_dir, "{}.png".format(frame_id))))
            h, w = label.shape
            N = h * w
            label = nyu40_to_scannet20(label)
            num_labelled_1 = np.count_nonzero(label > 0)
            label[label > 20] = 0
            num_labelled_2 = np.count_nonzero(label > 0)
            # print(num_labelled_1, num_labelled_2, num_labelled_2 / N)
            if (num_labelled_2 / N) < 0.1:
                continue

        valid_frame_ids.append(frame_id)

    # [N,]
    valid_frame_ids = np.asarray(valid_frame_ids)

    with open(os.path.join(save_dir, "{}.txt".format(scene)), "w") as f:
        for frame_id in valid_frame_ids:
            things_to_write = "{}\n".format(frame_id)
            f.write(things_to_write)

    return len(valid_frame_ids)


def create_valid_frames_lists(scannet_root, save_files_root, filter=False, skip=20):
    scene_files = ["configs/scannetv2_train.txt", "configs/scannetv2_val.txt"]
    suffix = "valid_frames"
    save_files_dir = os.path.join(save_file_root), "skip_{}/{}".format(skip, suffix)
    os.makedirs(save_files_dir, exist_ok=True)

    all_scenes_list = []
    for scene_file in scene_files:
        with open(scene_file, "r") as f:
            scenes = f.readlines()
            for scene in scenes:
                scene = scene.strip()  # remove \n
                all_scenes_list.append(scene)

    for scene in tqdm(all_scenes_list):
        print("Processing {}".format(scene))
        create_valid_frame_list(scannet_root, scene, save_files_dir, skip=skip, filter=filter)


def create_all_frame_list(scannet_dir, scene, save_dir, skip=20):
    """
    create frame list for all frames with skip interval
    :param scannet_dir:
    :param scene:
    :param save_dir:
    :return:
    """
    pose_dir = os.path.join(scannet_dir, scene, "pose")
    n_frames = len(os.listdir(pose_dir))

    # step 1: get valid frames
    frame_ids = list(range(0, n_frames, skip))

    with open(os.path.join(save_dir, "{}.txt".format(scene)), "w") as f:
        for frame_id in frame_ids:
            things_to_write = "{}\n".format(frame_id)
            f.write(things_to_write)

    return len(frame_ids)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_root", required=True)
    parser.add_argument("--save_files_root", type=str, default="./image_pairs/multiview")
    args = parser.parse_args()

    scannet_root = args.scannet_root
    save_file_root = args.save_files_root
    create_fragments_batched(scannet_root, save_file_root, filter=False, n_views=3, step=1, skip=20)
    create_fragments_causal(scannet_root, save_file_root, filter=False, n_views=3, step=1, skip=20)
    create_valid_frames_lists(scannet_root, save_file_root, filter=False,  skip=20)
