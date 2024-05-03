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

import torch
import torch.nn.functional as F
import numpy as np
import plyfile
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
)
from tqdm import tqdm

from config import load_yaml
from dataio.scannet import ScannetMultiViewDataset
from dataio.slamcore import SLAMcoreMultiViewDataset
from dataio.utils import color_encoding_scannet20, vert_label_to_color, get_scene_list, read_ply
from metric.iou import IoU3D
from networks.rend_utils import project_pcd
from train_lpn import get_model


def load_sequence(dataset_type, dataset_root, scene, skip=10, H=480, W=640, n_views=3, step=1):
    assert dataset_type in ["scannet", "scannet_test", "slamcore"], "Unknown dataset type!!!"
    if "scannet" in dataset_type:
        scene_list = [scene]
        sequence = ScannetMultiViewDataset(dataset_root,
                                           scene_list,
                                           phase="test",
                                           skip=skip,
                                           window_size=n_views,
                                           step=step,
                                           load_label=False,
                                           data_aug=False,
                                           clean_data=False,
                                           H=H, W=W,
                                           load_all=True)
    else:
        sequence = SLAMcoreMultiViewDataset(dataset_root,
                                            scene,
                                            skip=skip,
                                            window_size=n_views,
                                            H=H, W=W,
                                            load_all=True)
    return sequence


# label mesh
@torch.no_grad()
def label_one_scene(model,
                    scene,
                    sequence,
                    mesh_file,
                    save_root_dir,
                    H=480,
                    W=640,
                    fusion_mode="bayesian",
                    eps=0.05,
                    skip_exist=True,
                    device=torch.device("cuda")):
    """
    :return:
    """

    label_save_dir = os.path.join(save_root_dir, scene)
    os.makedirs(label_save_dir, exist_ok=True)
    saved_mesh_file = os.path.join(label_save_dir, "labelled_mesh_{}_{}.ply".format(scene, fusion_mode))
    if skip_exist and os.path.exists(saved_mesh_file):
        print("Already labelled! Skipping...")
        return

    mesh_data = read_ply(mesh_file)
    verts_np, faces_np = mesh_data["verts"], mesh_data["faces"]
    verts = torch.from_numpy(verts_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)

    # For now only use vertex-wise: simple projection
    V, _ = verts.shape
    vert_probs = torch.ones(V, 21, device=device) / 21.
    weights = torch.zeros(V, device=device)
    vert_highest_prob = torch.zeros(V, 1, device=device)
    vert_highest_prob_label = torch.zeros(V, 1, device=device).long()
    for i in tqdm(range(len(sequence))):
        frame = sequence[i]
        c2w, w2c, K, rgb, depth, frame_id = frame["c2w"].to(device), \
            frame["w2c"].to(device), \
            frame["K_depth"].to(device), \
            frame["rgb"].to(device), \
            frame["depth"].to(device), \
            frame["frame_id"]

        # print("Processing frame: {}".format(frame_id))
        if torch.isnan(c2w).any().item() or torch.isinf(c2w).any().item():
            # print("Skipping frame: {}".format(frame_id))
            continue

        uv_norm, valid_mask = project_pcd(verts.squeeze(), K, H, W, depth=depth.squeeze()[0], crop=0, w2c=w2c[0], eps=eps)
        valid_uv_norm = uv_norm[valid_mask, :]
        # forward pass
        with torch.no_grad():
            result = model(rgb.unsqueeze(0), depth.unsqueeze(0), K.unsqueeze(0), c2w.unsqueeze(0), w2c.unsqueeze(0))

        class_logit = result["out"][0]  # [21, H_test, W_test]
        class_prob = F.softmax(class_logit, dim=0)
        likelihood = F.grid_sample(class_prob.unsqueeze(0), valid_uv_norm.unsqueeze(0).unsqueeze(0), align_corners=False, padding_mode="border").squeeze().t()  # [N_valid, 21]

        if fusion_mode == "bayesian":
            p_post = vert_probs[valid_mask, :] * likelihood
            vert_probs[valid_mask, :] = p_post / torch.sum(p_post, dim=-1, keepdim=True)  # normalize
        elif fusion_mode == "average":
            vert_probs[valid_mask, :] = (weights[valid_mask].unsqueeze(-1) * vert_probs[valid_mask, :] + likelihood) / (weights[valid_mask].unsqueeze(-1) + 1.)
            weights[valid_mask] += 1.
        elif fusion_mode == "replacement":
            vert_probs[valid_mask, :] = likelihood
        elif fusion_mode == "highest_prob":
            highest_likelihood = torch.max(likelihood, dim=-1, keepdim=True)[0]  # [N_valid, 1]
            highest_likelihood_label = torch.argmax(likelihood, dim=-1, keepdim=True)  # [N_valid, 1]
            update_mask = (highest_likelihood > vert_highest_prob[valid_mask, :]).float()
            vert_highest_prob_label[valid_mask, :] = (1 - update_mask.long()) * vert_highest_prob_label[valid_mask, :] + update_mask.long() * highest_likelihood_label
            vert_highest_prob[valid_mask, :] = (1 - update_mask) * vert_highest_prob[valid_mask, :] + update_mask * highest_likelihood
        else:
            raise NotImplementedError

    torch.save(vert_probs, os.path.join(label_save_dir, "class_prob_{}.pth".format(fusion_mode)))
    if fusion_mode == "highest_prob":
        vert_label = vert_highest_prob_label.squeeze().cpu().numpy()
    elif fusion_mode in ["replacement", "bayesian", "average"]:
        vert_label = vert_probs.argmax(1).cpu().numpy()  # [V]
    else:
        raise NotImplementedError
    torch.save(torch.from_numpy(vert_label).long(), os.path.join(label_save_dir, "class_label_{}.pth".format(fusion_mode)))
    colors = torch.from_numpy(vert_label_to_color(vert_label, color_encoding_scannet20).astype(np.float32) / 255.)
    tex = TexturesVertex(verts_features=colors[None])
    # Only accepts batched input: [B, V, 3], [B, F, 3]
    mesh = Meshes(verts=verts[None], faces=faces[None], textures=tex)
    # For some reason, colors_as_uint8=True is required to save texture
    IO().save_mesh(mesh, os.path.join(label_save_dir, "labelled_mesh_{}_{}.ply".format(scene, fusion_mode)), colors_as_uint8=True)


def label_meshes(model,
                 dataset_type,
                 dataset_root, 
                 scene_list,
                 mesh_name_suffix,
                 save_root_dir,
                 H=480,
                 W=640,
                 skip=10,
                 n_views=3,
                 step=1,
                 fusion_mode="bayesian",
                 eps=0.05,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """
    This will labell all scannet scenes of a split
    :return:
    """
    for scene in tqdm(scene_list):
        # create dataset here
        sequence = load_sequence(dataset_type, dataset_root, scene, H=H, W=W, skip=skip, n_views=n_views, step=step)
        if dataset_type == "slamcore":
            W = sequence.W_crop
        mesh_name = scene + mesh_name_suffix
        mesh_file = os.path.join(dataset_root, scene, mesh_name)
        # TODO: return tensors and meshes. Save here
        label_one_scene(model,
                        scene,
                        sequence,
                        mesh_file,
                        save_root_dir,
                        H=H,
                        W=W,
                        fusion_mode=fusion_mode,
                        eps=eps,
                        device=device)


def eval_meshes(dataset_root, scene_list, labelled_meshes_root_dir, save_txt_path, fusion_mode="bayesian",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    metric = IoU3D(21, ignore_index=0)
    for scene in tqdm(scene_list):
        label_pred = torch.load(os.path.join(labelled_meshes_root_dir, scene, "class_label_{}.pth".format(fusion_mode)), map_location=device)
        label_gt = torch.load(os.path.join(dataset_root, scene, "label.pth".format(scene)), map_location=device)
        metric.add(label_pred, label_gt)

    iou, miou = metric.value()
    with open(save_txt_path, "w") as f:
        f.write("-------------Labelled Mesh Result--------------\n")
        for key, class_iou in zip(color_encoding_scannet20.keys(), iou):
            f.write("{0}: {1:.4f}\n".format(key, class_iou))
        f.write("Mean IoU: {}\n".format(miou))
        f.write("Mean IoU (ignore unlabelled): {}".format(iou[1:].mean()))


def bayesian_label(model, dataset_type, dataset_root, scene_list, mesh_name_suffix, save_root_dir, skip=10, n_views=3, suffix="val", 
    eval=True, device=torch.device("cuda:0"), H=480, W=640):
    # slamcore depth too bad, have to increase eps
    if dataset_type == "slamcore":
        eps = 0.20
    else:
        eps = 0.05

    label_meshes(model, dataset_type, dataset_root, scene_list, mesh_name_suffix,
                 save_root_dir,
                 skip=skip,
                 n_views=n_views,
                 step=1,
                 fusion_mode="bayesian",
                 eps=eps,
                 device=device,
                 H=H, W=W)
    if eval:
        save_result_file = os.path.join(save_root_dir, "bayesian_label_results_{}.txt".format(suffix))
        eval_meshes(dataset_root, scene_list, save_root_dir, save_result_file, fusion_mode="bayesian", device=device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True, help="DIR to the LPN log")
    parser.add_argument("--dataset_type", type=str, default="scannet")
    parser.add_argument("--dataset_root", type=str, required=True, help="DIR to dataset root")
    parser.add_argument('--save_dir', type=str, required=True, help="DIR to the labelled meshes")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=19)
    parser.add_argument("--n_views", type=int, default=3)
    parser.add_argument("--skip", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--H", type=int, default=480)
    parser.add_argument("--W", type=int, default=640)
    args = parser.parse_args()

    # load model
    log_dir = args.log_dir
    device = torch.device("cuda:{}".format(args.gpu))
    cfg = load_yaml(os.path.join(log_dir, "config.yaml"))
    cfg.window_size = args.n_views
    cfg.log_dir = log_dir
    chkpt_path = os.path.join(log_dir, "checkpoints/chkpt-{}.pth".format(args.epoch))
    model = get_model(cfg, device=device)
    pretrained_state_dict = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(pretrained_state_dict["state_dict"])
    model.eval()

    if args.dataset_type == "scannet":
        # label all scenes in train/val split and save under save_root_dir
        os.makedirs(args.save_dir, exist_ok=True)
        scene_split_train = "configs/scannetv2_train.txt"
        train_list = get_scene_list(scene_split_train)
        scene_split_val = "configs/scannetv2_val.txt"
        val_list = get_scene_list(scene_split_val)
        mesh_name_suffix = "_vh_clean_2.labels.ply"
        bayesian_label(model, args.dataset_type, args.dataset_root, train_list, mesh_name_suffix, args.save_dir, skip=args.skip, n_views=args.n_views, suffix="train", eval=False)
        bayesian_label(model, args.dataset_type, args.dataset_root, val_list, mesh_name_suffix, args.save_dir, skip=args.skip, n_views=args.n_views, suffix="val", eval=True)
    elif args.dataset_type == "scannet_test":
        os.makedirs(args.save_dir, exist_ok=True)
        scene_split_test = "configs/scannetv2_test.txt"
        scene_list = get_scene_list(scene_split_test)
        mesh_name_suffix = "_vh_clean_2.ply"
        bayesian_label(model, args.dataset_type, args.dataset_root, scene_list, mesh_name_suffix, args.save_dir, skip=args.skip, n_views=args.n_views, eval=False)
    elif args.dataset_type == "slamcore":
        os.makedirs(args.save_dir, exist_ok=True)
        scene_list = get_scene_list("configs/slamcore.txt")
        mesh_name_suffix = "_clean.labels.ply"
        bayesian_label(model, args.dataset_type, args.dataset_root, scene_list, mesh_name_suffix, args.save_dir, skip=args.skip, n_views=args.n_views, 
            suffix="slamcore_test", eval=True, H=480, W=848)
    else:
        raise NotImplementedError
