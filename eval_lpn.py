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
import imageio
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from config import load_yaml
from dataio.scannet import ScannetMultiViewDataset
from dataio.utils import create_label_image, color_encoding_scannet20, get_scene_list
from train_lpn import get_model, LPNTester
from eval_lpn_bayesian_label import load_sequence
from metric.iou import IoU


"""
Dataset
"""

@torch.no_grad()
def inference_one_scene(model, dataset_type, dataset_root, scene, save_root_dir, skip=20, n_views=3,
                        step=1, H=480, W=640, device=torch.device("cuda:0")):
    save_dir = os.path.join(save_root_dir, scene)
    os.makedirs(save_dir, exist_ok=True)
    sequence = load_sequence(dataset_type, dataset_root, scene, H=H, W=W, skip=skip, n_views=n_views, step=step)
    for i in tqdm(range(len(sequence))):
        frame = sequence[i]
        c2w, w2c, K, rgb, depth, frame_id = frame["c2w"].to(device), \
                                            frame["w2c"].to(device), \
                                            frame["K_depth"].to(device), \
                                            frame["rgb"].to(device), \
                                            frame["depth"].to(device), \
                                            frame["frame_id"]
        with torch.no_grad():
            result = model(rgb.unsqueeze(0), depth.unsqueeze(0), K.unsqueeze(0), c2w.unsqueeze(0), w2c.unsqueeze(0))

        # predicted label
        output = result["out"][0]
        label_pred = output.argmax(0).cpu().numpy()
        label_pred_image = create_label_image(label_pred, color_encoding_scannet20)
        rgb_raw = frame["rgb_raw"][0].cpu().numpy()

        # save result
        image_to_save = np.zeros((H, W * 2, 3))
        image_to_save[:, :W, :] = rgb_raw
        image_to_save[:, W:2 * W, :] = label_pred_image
        imageio.imwrite(os.path.join(save_dir, "{}.png".format(frame_id)), image_to_save.astype(np.uint8))


def evaluate(cfg, model, dataset_root, scene_list, result_save_path,
             skip=20, n_views=3, step=1, H=480, W=640,
             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    # create datasets and data loaders
    val_set = ScannetMultiViewDataset(dataset_root,
                                      scene_list,
                                      phase="test",
                                      skip=skip,
                                      window_size=n_views,
                                      step=step,
                                      data_aug=False,
                                      clean_data=False,
                                      H=H,
                                      W=W,
                                      load_all=True,
                                      load_label=True)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    class_encoding = val_set.color_encoding
    num_classes = len(class_encoding)
    class_weights = np.loadtxt(cfg.class_weights_file)
    # create IoU class
    class_weights = torch.from_numpy(class_weights).float().to(device)
    metric = IoU(num_classes, ignore_index=0)
    # criterion = nn.NLLLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    val = LPNTester(cfg, model, val_loader, criterion, metric, device)
    result_dict = val.run_epoch(cfg.print_every)
    loss, (iou, miou) = result_dict["loss"], result_dict["metric"]

    with open(result_save_path, "w") as f:
        f.write("-------------2D Evaluation Result--------------\n")
        for key, class_iou in zip(class_encoding.keys(), iou):
            f.write("{0}: {1:.4f}\n".format(key, class_iou))
        f.write("Mean IoU: {}\n\n".format(miou))


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="log_dir of trained LPN")
    parser.add_argument("--dataset_type", type=str, default="scannet")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    eval_parser = parser.add_mutually_exclusive_group(required=True)
    eval_parser.add_argument("--eval", dest="eval", action="store_true")
    eval_parser.add_argument("--inference", dest="eval", action="store_false")
    parser.set_defaults(eval=True)
    parser.add_argument("--scene", type=str, help="scene name, optional")
    parser.add_argument("--epoch", default=19)
    parser.add_argument("--n_views", type=int, default=3)
    parser.add_argument("--skip", type=int, default=20)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--H", type=int, default=480)
    parser.add_argument("--W", type=int, default=640)
    args = parser.parse_args()

    # Load LPN model
    cfg = load_yaml(os.path.join(args.log_dir, "config.yaml"))
    cfg.window_size = args.n_views  # window_size could be changed at run-time for LPN model
    chkpt_path = os.path.join(args.log_dir, "checkpoints/chkpt-{}.pth".format(args.epoch))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg, device=device)
    pretrained_state_dict = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(pretrained_state_dict["state_dict"])
    model.eval()

    dataset_type = args.dataset_type
    dataset_root = args.dataset_root
    if dataset_type == "scannet":
        scene_split_file = "configs/scannetv2_val.txt"
    elif dataset_type == "scannet_test":
        scene_split_file = "configs/scannetv2_test.txt"
    elif dataset_type == "slamcore":
        scene_split_file = "configs/slamcore.txt"
    else:
        raise NotImplementedError

    os.makedirs(args.save_dir, exist_ok=True)
    scene_list = get_scene_list(scene_split_file)
    if args.eval:
        assert args.dataset_type == "scannet", "Only scannet_val has 2D GT labels!!!"
        result_save_path = os.path.join(args.save_dir, "result_lpn_2d.txt")
        evaluate(cfg, model, dataset_root, scene_list, result_save_path,
                 skip=args.skip, n_views=args.n_views, step=args.step, H=args.H, W=args.W, device=device)
    elif args.scene is None:  # otherwise inference a specific scene or all the scenes
        for scene in tqdm(scene_list):
            inference_one_scene(model, dataset_type, dataset_root, scene, args.save_dir,
                                skip=args.skip, n_views=args.n_views, step=args.step, H=args.H, W=args.W, device=device)
    else:
        inference_one_scene(model, dataset_type, dataset_root, args.scene, args.save_dir,
                            skip=args.skip, n_views=args.n_views, step=args.step, H=args.H, W=args.W, device=device)


if __name__ == "__main__":
    eval()
