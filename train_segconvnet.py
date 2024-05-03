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
import time
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import config
from networks.SegConvNet import SegConvNet
from dataio.segment_dataset import SegmentDatasetTrain, SegmentDatasetTest
from dataio.utils import get_scene_list
from config import save_config
from metric.iou import IoU3D


def load_data(cfg):
    train_list = get_scene_list(cfg.train_file)
    train_set = SegmentDatasetTrain(cfg.label_fusion_dir,
                                    cfg.segment_suffix,
                                    train_list,
                                    k=cfg.k,
                                    data_aug=cfg.data_aug,
                                    feat_type=cfg.feat_type,
                                    use_xyz=cfg.use_xyz
    )
    train_loader = train_set.get_dataloader(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_custom_sampler=cfg.use_custom_sampler,
        drop_last=True
    )

    val_list = get_scene_list(cfg.val_file)
    val_set = SegmentDatasetTest(cfg.label_fusion_dir,
                                 cfg.segment_suffix,
                                 val_list,
                                 k=cfg.k,
                                 feat_type=cfg.feat_type,
                                 use_xyz=cfg.use_xyz
    )
    val_loader = val_set.get_dataloader()

    class_encoding = train_set.color_encoding
    num_classes = len(class_encoding)

    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # load pre-computed class weights
    class_weights = np.loadtxt(cfg.class_weights_file)
    # ignore_index = list(class_encoding).index('unlabeled')
    # class_weights[ignore_index] = 0.0

    return train_loader, val_loader, class_weights, class_encoding


def create_logdir(cfg):
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.event_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)


class Train:
    def __init__(self, model, data_loader, optim, criterion, metric, device, lr_scheduler=None, global_step=0, writer=None):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.global_step = global_step
        self.writer = writer
        self.lr_scheduler = lr_scheduler

    def run_epoch(self, print_every=0):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        avgTime = 0.0
        numTimeSteps = 0
        for step, batch_data in enumerate(self.data_loader):
            startTime = time.time()

            # Get the inputs and labels
            xyz = batch_data["locs"].to(self.device)  # [1, N_seg_batch, 3]
            cov = batch_data["covs"].to(self.device)  # [1, N_seg_batch, 3, 3]
            feat = batch_data["feats"].to(self.device)  # [1, N_seg_batch, C]
            knn_indices = batch_data["knn_indices"].to(self.device)  # [N_seg_batch, K]
            label = batch_data["labels"].long().to(self.device)  # [1, N_seg_batch]
            B, N, C = feat.shape

            # Forward propagation
            out = self.model(xyz, cov, feat, knn_indices)  # [1, n_classes, N_seg_batch]

            # Loss computation CrossEntropy
            loss = self.criterion(out, label)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # lr scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
                lr = self.lr_scheduler.get_lr()[0]
            else:
                lr = None

            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            # Keep track of the evaluation metric
            self.metric.add(
                out.detach().permute(0, 2, 1).view(B * N, -1),  # [N_seg_batch, n_classes]
                label.detach().view(B * N)  # [N_seg_batch,]
            )
            endTime = time.time()
            avgTime += (endTime - startTime)
            numTimeSteps += 1

            if print_every > 0 and (step % print_every == 0):
                print("[Step: %d/%d (%3.2f ms)] Iteration loss: %.4f" % (step, len(self.data_loader), \
                                                                         1000*(avgTime / (numTimeSteps if numTimeSteps>0 else 1)), loss.item()))
                if self.writer:
                    self.writer.add_scalar("Train/loss", loss.item(), self.global_step)
                    if lr:
                        self.writer.add_scalar("Train/LR", lr, self.global_step)

                numTimeSteps = 0
                avgTime = 0.

            self.global_step += 1

            torch.cuda.empty_cache()

        return epoch_loss / len(self.data_loader), self.metric.value()


class Test:
    def __init__(self, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device

    @torch.no_grad()
    def run_epoch(self, print_every=0):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.eval()
        epoch_loss = []
        self.metric.reset()
        avgTime = 0.0
        numTimeSteps = 0
        for step, batch_data in enumerate(self.data_loader):
            startTime = time.time()

            # Get the inputs and labels
            xyz = batch_data["locs"].to(self.device)  # [1, N_seg_batch, 3]
            cov = batch_data["covs"].to(self.device)  # [1, N_seg_batch, 3, 3]
            feat = batch_data["feats"].to(self.device)  # [1, N_seg_batch, C]
            knn_indices = batch_data["knn_indices"].to(self.device)  # [N_seg_batch, K]
            label = batch_data["labels"].long().to(self.device)  # [1, N_seg_batch]
            B, N, C = feat.shape

            # Forward propagation
            out = self.model(xyz, cov, feat, knn_indices)  # [1, n_classes, N_seg_batch]

            # Loss computation
            # CrossEntropy
            loss = self.criterion(out, label)

            # Keep track of loss for current epoch
            epoch_loss.append(loss.item())

            # Keep track of the evaluation metric
            self.metric.add(
                out.detach().permute(0, 2, 1).view(B * N, -1),  # [N_seg_batch, n_classes]
                label.detach().view(B * N)  # [N_seg_batch,]
            )
            endTime = time.time()
            avgTime += (endTime - startTime)
            numTimeSteps += 1

            if print_every > 0 and (step % print_every == 0):
                print("[Step: %d/%d (%3.2f ms)] Iteration loss: %.4f" % (step, len(self.data_loader), \
                                                                         1000*(avgTime / (numTimeSteps if numTimeSteps>0 else 1)), loss.item()))

                numTimeSteps = 0
                avgTime = 0.

        return np.nanmean(np.array(epoch_loss)), self.metric.value()


def update_cfg(cfg, args):
    cfg.label_fusion_dir = args.label_fusion_dir
    cfg.segment_suffix = args.segment_suffix
    cfg.log_dir = args.log_dir
    cfg.event_dir = os.path.join(cfg.log_dir, "events")
    cfg.save_dir = os.path.join(cfg.log_dir, "checkpoints")

    for k, v in vars(args).items():
        if v is not None:
            cfg.setdefault(k, v)

    return cfg


def get_model(cfg):
    in_dim = 30 if cfg.use_xyz else 21
    model = SegConvNet(input_feat_dim=in_dim,
                       num_classes=21,
                       dropout_p=cfg.dropout_p,
                       weight_in=cfg.setdefault("weight_in", "xyz"),
                       classifier_hidden_dims=cfg.setdefault("classifier_hidden_dims", [128, 64]))
    return model


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="config file")
    parser.add_argument("--log_dir", type=str, required=True, help="path to save log")
    parser.add_argument("--label_fusion_dir", type=str, required=True, help="path to bayesian-fused scens")
    parser.add_argument("--segment_suffix", type=str, required=True, help="segment suffix")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.set_defaults(eval=False)

    # setting of training SegConvNet
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--use_custom_sampler", dest="use_custom_sampler", action="store_true")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--scheduler", type=str, help="[step, onecycle]")
    parser.add_argument("--lr_decay_epochs", type=int, default=30)
    parser.add_argument("--lr_decay", type=float, default=0.2)
    parser.add_argument("--div_factor", type=float, default=1.0)
    parser.add_argument("--pct_start", type=float, default=0.05)
    parser.add_argument("--final_div_factor", type=float, default=1000.0)
    parser.add_argument("--anneal_strategy", type=str, default="cos")
    parser.add_argument("--dropout_p", type=float, default=-1.0)
    parser.add_argument("--feat_type", type=str, default="prob")
    parser.add_argument("--no_xyz", dest="use_xyz", action="store_false")
    parser.add_argument("--class_weights_file", type=str, default="configs/class_weights_scannet20_valid.txt")
    parser.add_argument("--eval_epoch", type=int, default=99)
    parser.add_argument("--k", type=int, default=10)
    parser.set_defaults(use_custom_sampler=False, use_xyz=True)
    # data_aug_parser = parser.add_mutually_exclusive_group(required=False)
    # data_aug_parser.add_argument("--data_aug", dest="data_aug", action="store_true")
    # data_aug_parser.add_argument("--no_data_aug", dest="data_aug", action="store_false")
    parser.set_defaults(data_aug=True)
    args = parser.parse_args()

    cfg = config.load_yaml(args.config)
    cfg = update_cfg(cfg, args)
    create_logdir(cfg)
    print(cfg)
    save_config(cfg, os.path.join(cfg.log_dir, "config.yaml"))
    shutil.copy("networks/LatentPriorNetwork.py", os.path.join(cfg.log_dir, "LatentPriorNetwork.py"))
    shutil.copy("networks/SegConvNet.py", os.path.join(cfg.log_dir, "SegConvNet.py"))
    writer = SummaryWriter(log_dir=cfg.event_dir)

    # create datasets and data loaders
    train_loader, val_loader, class_weights, class_encoding = load_data(cfg)
    num_classes = len(class_encoding)

    # create model and optimizer
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = get_model(cfg)
    model.to(device)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction=cfg.loss_reduction)
    criterion_val = nn.CrossEntropyLoss(weight=class_weights, reduction=cfg.loss_reduction_test)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta0, cfg.beta1),
        weight_decay=cfg.weight_decay
    )
    # Learning rate decay scheduler
    if cfg.scheduler == "none":
        lr_updater = None
    elif cfg.scheduler == "step":
        lr_updater = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_decay_epochs * len(train_loader), cfg.lr_decay)
    elif cfg.scheduler == 'onecycle':
        # len(data_loader) == number of batches
        # len(dataset) == number of data points
        total_steps = cfg.epochs * len(train_loader)
        lr_updater = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            total_steps=total_steps,
            div_factor=cfg.div_factor,  # initial
            pct_start=cfg.pct_start,
            anneal_strategy=cfg.anneal_strategy,
            final_div_factor=cfg.final_div_factor
        )
    else:
        raise NotImplementedError

    # Load checkpoint if any
    checkpoints = [p for p in os.listdir(cfg.save_dir) if not p.endswith("best.pth")]
    if len(checkpoints) > 0:
        chkpt_path = os.path.join(cfg.save_dir, sorted(checkpoints, key=lambda x: int(x[6:-4]))[-1])
        print("Resume training from {}".format(chkpt_path))
        chkpt = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(chkpt["state_dict"])
        optimizer.load_state_dict(chkpt["optimizer"])
        if lr_updater is not None:
            lr_updater.load_state_dict(chkpt["lr_scheduler"])
        start_epoch = int(chkpt["epoch"]) + 1
        start_iter = chkpt["n_iter"] + 1
    else:
        print("Training from scratch...")
        start_epoch = 0
        start_iter = 0

    # Evaluation metric
    ignore_index = list(class_encoding).index('unlabeled')
    metric = IoU3D(num_classes, ignore_index=ignore_index)
    train = Train(model, train_loader, optimizer, criterion, metric, device,
                  lr_scheduler=lr_updater, global_step=start_iter, writer=writer)
    val = Test(model, val_loader, criterion_val, metric, device)
    best_miou = 0.
    for epoch in range(start_epoch, cfg.epochs):
        epoch_loss, (iou, miou) = train.run_epoch(cfg.print_every)
        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))
        writer.add_scalar("Train/epoch_loss", epoch_loss, epoch)
        writer.add_scalar("Train/miou", miou, epoch)

        # validate
        if (epoch + 1) % cfg.validate_every == 0 or epoch + 1 == cfg.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            val_loss, (iou, miou) = val.run_epoch(cfg.print_every)
            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, val_loss, miou))
            writer.add_scalar("Val/epoch_loss", val_loss, epoch)
            writer.add_scalar("Val/miou", miou, epoch)
            for key, class_iou in zip(class_encoding.keys(), iou):
                print("{0}: {1:.4f}".format(key, class_iou))

        # save current best
        if miou > best_miou:
            checkpoint = {
                'epoch': epoch,
                'n_iter': train.global_step,
                'miou': miou,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if lr_updater is not None:
                checkpoint["lr_scheduler"] = lr_updater.state_dict()
            torch.save(checkpoint, os.path.join(cfg.save_dir, "chkpt-best.pth"))
            best_miou = miou

        # save checkpoint
        if epoch + 1 == cfg.epochs or (epoch + 1) % cfg.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'n_iter': train.global_step,
                'miou': miou,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if lr_updater is not None:
                checkpoint["lr_scheduler"] = lr_updater.state_dict()
            torch.save(checkpoint, os.path.join(cfg.save_dir, "chkpt-{}.pth".format(epoch)))

    print("Best validation miou: {}".format(best_miou))


if __name__ == "__main__":
    train()
