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

import trimesh
from tqdm import tqdm

from config import load_yaml
from dataio.utils import get_scene_list, color_encoding_scannet20, vert_label_to_color, scannet20_to_nyu40, read_ply
from dataio.segment_dataset import SegmentDatasetTest
from metric.iou import IoU3D
from train_segconvnet import get_model


@torch.no_grad()
def label_one_mesh(model, cfg, dataset_type, dataset_root, scene, mesh_name,
                   label_fusion_dir, segment_suffix, save_root_dir,
                   device=torch.device("cuda")):
    """
    :return:
    """

    # Same dataset for both dataset types
    scene_segments = SegmentDatasetTest(label_fusion_dir,
                                        segment_suffix,
                                        [scene],
                                        dataset_type=dataset_type,
                                        k=cfg.k,
                                        use_xyz=cfg.use_xyz,
                                        feat_type=cfg.feat_type,
                                        load_label=False)
    
    gt_mesh_path = os.path.join(dataset_root, scene, mesh_name)
    assert os.path.exists(gt_mesh_path), "GT mesh doesn't exist!!!"
    save_dir = os.path.join(save_root_dir, scene)
    save_label_pth = os.path.join(save_dir, "verts_label.pth")
    if os.path.exists(save_label_pth):
        return
    
    os.makedirs(save_dir, exist_ok=True)
    # only 1 data
    data_loader = scene_segments.get_dataloader()
    data = next(iter(data_loader))
    xyz = data["locs"].to(device)  # [1, N_seg_batch, 3]
    cov = data["covs"].to(device)  # [1, N_seg_batch, 3, 3]
    feat = data["feats"].to(device)  # [1, N_seg_batch, C]
    knn_indices = data["knn_indices"].to(device)  # [N_seg_batch, K]

    # Forward propagation
    out = model(xyz, cov, feat, knn_indices).squeeze()  # [n_classes, N_seg]
    seg_label_pred = torch.argmax(out, dim=0).cpu().numpy()  # [N_seg,]

    # load segment.pth
    scene_segment_dir = os.path.join(dataset_root, scene, segment_suffix)
    segments = torch.load(os.path.join(scene_segment_dir, "segments.pth")).cpu().numpy()  # [V,] with N_seg_all ids
    
    # initialize with bayesian labbelled results
    scene_label_fusion_dir = os.path.join(label_fusion_dir, scene)
    verts_label_pred = torch.load(os.path.join(scene_label_fusion_dir, "class_label_bayesian.pth")).cpu().numpy()  # [V,]
    assert len(segments) == len(verts_label_pred), "Number of vertices mismatch!!!"
    seg_ids_all = np.unique(segments)  # [N_seg_all,]
    valid_segments_mask_path = os.path.join(scene_label_fusion_dir, segment_suffix, "valid_segments_mask.pth")
    if os.path.exists(valid_segments_mask_path):
        valid_segments_mask = torch.load(valid_segments_mask_path).cpu().numpy()  # [N_seg_all,]
        seg_ids = seg_ids_all[valid_segments_mask]  # [N_seg]
    else:
        seg_ids = seg_ids_all
    assert len(seg_label_pred) == len(seg_ids), "Number of segments mismatch!!! {}, {}, {}".format(len(seg_label_pred), len(seg_ids), len(seg_ids_all))

    # 0 to N_seg - 1
    for i, seg_id in enumerate(seg_ids):
        # in very rare cases, the seg_ids is not range(N_seg)...
        seg_mask = (segments == seg_id)
        verts_label_pred[seg_mask] = seg_label_pred[i]

    torch.save(torch.from_numpy(verts_label_pred), save_label_pth)
    
    # save labelled mesh
    mesh_data = read_ply(gt_mesh_path)
    verts, faces = mesh_data["verts"], mesh_data["faces"]
    vert_colors = vert_label_to_color(verts_label_pred, color_encoding_scannet20)
    assert verts.shape[0] == verts_label_pred.shape[0], "Number of vertices doesn't match!!!"
    mesh_save = trimesh.Trimesh(verts, faces, vertex_colors=vert_colors)
    mesh_save.export(save_label_pth.replace(".pth", ".ply"))


def apply_3dconv_to_meshes(log_dir, dataset_type, dataset_root, scene_list, mesh_name_suffix,
                           label_fusion_dir, segment_suffix, save_dir, epoch=99):
    cfg_file = os.path.join(log_dir, "config.yaml")
    cfg = load_yaml(cfg_file)
    device = torch.device("cuda")
    model = get_model(cfg)
    model.to(device)
    chkpt = torch.load(os.path.join(log_dir, "checkpoints/chkpt-{}.pth".format(epoch)), map_location=device)
    model.load_state_dict(chkpt["state_dict"])
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    for scene in tqdm(scene_list):
        label_one_mesh(model, cfg, dataset_type, dataset_root, scene, scene + mesh_name_suffix,
                       label_fusion_dir, segment_suffix, save_dir, device=device)


def eval_meshes(dataset_root, pred_meshes_root, scene_list, pred_label_name="verts_label.pth", suffix="final_val"):
    metric = IoU3D(21, ignore_index=0)
    for scene in scene_list:
        gt_label = torch.load(os.path.join(dataset_root, scene, "label.pth"))
        pred_label = torch.load(os.path.join(pred_meshes_root, scene, pred_label_name))
        metric.add(pred_label, gt_label)

    iou, miou = metric.value()
    with open(os.path.join(pred_meshes_root, "result_{}.txt".format(suffix)), "w") as f:
        f.write("-------------Labelled Meshes Results--------------\n")
        for key, class_iou in zip(color_encoding_scannet20.keys(), iou):
            f.write("{0}: {1:.4f}\n".format(key, class_iou))
        f.write("Mean IoU: {}\n".format(miou))
        f.write("Mean IoU (ignore unlabelled): {}".format(iou[1:].mean()))


def convert_label_pred_to_nyu40_txt(log_dir, epoch=99, k=10):
    split = "val"
    label_dir = os.path.join(log_dir, "labelled_meshes/scannet/{}/k={}/{}".format(epoch, k, split))
    save_dir = os.path.join(log_dir, "labelled_meshes/scannet/{}/k={}/{}_nyu40_txt".format(epoch, k, split))
    os.makedirs(save_dir, exist_ok=True)

    cfg_file = os.path.join(log_dir, "config.yaml")
    cfg = load_yaml(cfg_file)
    scene_list = get_scene_list(cfg.val_file)
    for scene in scene_list:
        label_scannet20 = torch.load(os.path.join(label_dir, "{}/verts_label.pth".format(scene))).cpu().numpy().astype(np.int32)
        label_nyu40 = scannet20_to_nyu40(label_scannet20)
        np.savetxt(os.path.join(save_dir, "{}.txt".format(scene)), label_nyu40, fmt="%d")


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="log_dir of SegConvNet")
    parser.add_argument("--dataset_type", type=str, default="scannet")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--label_fusion_dir", type=str, required=True)
    parser.add_argument("--segment_suffix", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epoch", default=99)
    
    args = parser.parse_args()
    if args.dataset_type == "scannet":
        mesh_name_suffix = "_vh_clean_2.labels.ply"
        scene_split_file = "configs/scannetv2_val.txt"
    elif args.dataset_type == "scannet_test":
        mesh_name_suffix = "_vh_clean_2.ply"
        scene_split_file = "configs/scannetv2_test.txt"
    elif args.dataset_type == "slamcore":
        mesh_name_suffix = "_clean.labels.ply"
        scene_split_file = "configs/slamcore.txt"
    else:
        raise NotImplementedError
    
    scene_list = get_scene_list(scene_split_file)
    apply_3dconv_to_meshes(args.log_dir, args.dataset_type, args.dataset_root, scene_list, mesh_name_suffix, 
                           args.label_fusion_dir, args.segment_suffix, args.save_dir, epoch=args.epoch)
    eval_meshes(args.dataset_root, args.save_dir, scene_list, suffix="final_val")
    eval_meshes(args.dataset_root, args.label_fusion_dir, scene_list, "{}/label_major_vote_pred.pth".format(args.segment_suffix), suffix="major_vote_val")


if __name__ == "__main__":
    eval()
