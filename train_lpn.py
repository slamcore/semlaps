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

import copy
import os
import argparse
import shutil
import time
import imageio
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from config import load_yaml, save_config
from dataio.scannet import ScannetMultiViewDataset
from dataio.utils import create_label_image, color_encoding_scannet20, get_scene_list
from dataio.transforms_multiview import PIXEL_MEAN, PIXEL_STD
from networks.LatentPriorNetwork import LatentPriorNetwork
from metric.iou import IoU


def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()


def load_dataset(cfg):
    train_list = get_scene_list(cfg.train_file)
    train_set = ScannetMultiViewDataset(cfg.scannet_root,
                                        train_list,
                                        skip=cfg.skip,
                                        window_size=cfg.window_size,
                                        step=cfg.step,
                                        phase="train",
                                        data_aug=cfg.data_aug,
                                        depth_err=cfg.depth_err,
                                        clean_data=cfg.clean_data,
                                        H=cfg.H, W=cfg.W)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               pin_memory=False,
                                               drop_last=True,
                                               shuffle=True)

    val_list = get_scene_list(cfg.val_file)
    val_set = ScannetMultiViewDataset(cfg.scannet_root,
                                      val_list,
                                      skip=cfg.skip,
                                      window_size=cfg.window_size,
                                      step=cfg.step,
                                      phase="test",
                                      data_aug=False,
                                      clean_data=False,
                                      H=cfg.H, W=cfg.W)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=cfg.num_workers)

    class_encoding = train_set.color_encoding
    num_classes = len(class_encoding)
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))
    print("Loading data from: {}".format(cfg.scannet_root))
    print("Using {} workers for data-loading".format(cfg.num_workers))

    # load pre-computed class weights
    class_weights = np.loadtxt(cfg.class_weights_file)
    # shouldn't ignore unlabelled here!
    # ignore_index = list(class_encoding).index('unlabeled')
    # class_weights[ignore_index] = 0.0

    return (train_loader, val_loader), class_weights, class_encoding


def create_logdir(cfg):
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.event_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)


class LPNTrainer:
    """
    TODO: 
    """
    def __init__(self, cfg, model, data_loader, optim, criterion, metric, device, lr_scheduler=None, global_step=0, writer=None):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.global_step = global_step
        self.writer = writer
        self.lr_scheduler = lr_scheduler

        self.aux_loss = cfg.aux_loss
        self.weight_aux = cfg.weight_aux
        if self.aux_loss:
            print("Using auxiliary loss for training!")
            self.metric_aux = copy.deepcopy(metric)
        self.p_drop_depth = cfg.p_drop_depth

    def run_epoch(self, iteration_loss=0):
        """
        Runs an epoch of training.
        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        if self.aux_loss:
            self.metric_aux.reset()
        avgTime = 0.0
        numTimeSteps = 0
        for step, batch_data in enumerate(self.data_loader):
            startTime = get_time()

            # Get the inputs and labels
            rgb_inputs = batch_data["rgb"].to(self.device)  # [B, N, 3, H, W]
            depth_inputs = batch_data["depth"].to(self.device)  # [B, N, 1, H, W]
            labels = batch_data["label"].long().to(self.device)  # [B, N, H, W]
            labels_ref = labels[:, 0, ...]  # [B, H, W]
            bs, n_views, h, w = labels.shape
            labels_flatten = labels.view(bs * n_views, h, w)  # [BN, H, W]
            K = batch_data["K_depth"].to(self.device)  # [B, 3, 3]
            c2w = batch_data["c2w"].to(self.device)  # [B, N, 4, 4]
            w2c = batch_data["w2c"].to(self.device)  # [B, N, 4, 4]

            # Forward propagation
            result = self.model(rgb_inputs, depth_inputs, K, c2w, w2c, aux_loss=self.aux_loss, p_drop_depth=self.p_drop_depth)

            # Loss computation
            # loss = self.criterion(logit_pred, labels)
            logit_pred = result["out"]  # [B, C, H, W]
            loss = self.criterion(logit_pred, labels_ref)
            if self.aux_loss:
                logit_pred_aux = result["aux"]
                loss_aux = self.criterion(logit_pred_aux, labels_flatten)
                loss += self.weight_aux * loss_aux

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # lr scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            # get_last_lr()
            lr = self.lr_scheduler.get_lr()[0]

            # Keep track of the evaluation metric
            self.metric.add(logit_pred.detach(), labels_ref.detach())
            if self.aux_loss:
                self.metric_aux.add(logit_pred_aux.detach(), labels_flatten.detach())

            endTime = get_time()
            avgTime += (endTime - startTime)
            numTimeSteps += 1

            if iteration_loss > 0 and (step % iteration_loss == 0):
                print("[Step: %d/%d (%3.2f ms)] Iteration loss: %.4f" % (step, len(self.data_loader), \
                                                                         1000*(avgTime / (numTimeSteps if numTimeSteps>0 else 1)), loss.item()))
                if self.writer:
                    self.writer.add_scalar("Train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("Train/LR", lr, self.global_step)

                numTimeSteps = 0
                avgTime = 0.

            self.global_step += 1

        ret = {
            "loss": epoch_loss / len(self.data_loader),
            "metric": self.metric.value()
        }
        if self.aux_loss:
            ret["metric_aux"] = self.metric_aux.value()

        return ret


class LPNTester:
    """
    """
    def __init__(self, cfg, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.aux_loss = cfg.aux_loss
        self.weight_aux = cfg.weight_aux
        if self.aux_loss:
            print("Using auxiliary loss for validation!")
            self.metric_aux = copy.deepcopy(metric)
        self.log_dir = cfg.log_dir
        self.check_label_dir = os.path.join(self.log_dir, "check_label")

    @torch.no_grad()
    def run_epoch(self, iteration_loss=0, check_label=False):
        """
        """
        self.model.eval()
        epoch_loss = []
        self.metric.reset()
        if self.aux_loss:
            self.metric_aux.reset()
        avgTime = 0.0
        numTimeSteps = 0
        for step, batch_data in enumerate(self.data_loader):
            startTime = get_time()
            # Get the inputs and labels
            rgb_inputs = batch_data["rgb"].to(self.device)  # [B, N, 3, H, W]
            depth_inputs = batch_data["depth"].to(self.device)
            labels = batch_data["label"].long().to(self.device)  # [B, N, H, W]
            labels_ref = labels[:, 0, ...]  # [B, H, W]
            bs, n_views, h, w = labels.shape
            labels_flatten = labels.view(bs * n_views, h, w)  # [BN, H, W]
            K = batch_data["K_depth"].to(self.device)  # [B, 3, 3]
            c2w = batch_data["c2w"].to(self.device)  # [B, N, 4, 4]
            w2c = batch_data["w2c"].to(self.device)  # [B, N, 4, 4]

            H, W = rgb_inputs.shape[-2:]
            H_test, W_test = labels.shape[-2:]
            assert H == H_test and W == W_test, "validation label and input must have the same size!!!"

            # Loss computation
            # loss = self.criterion(logit_pred, labels)
            # Forward propagation
            result = self.model(rgb_inputs, depth_inputs, K, c2w, w2c, aux_loss=self.aux_loss)
            logit_pred = result["out"]
            loss = self.criterion(logit_pred, labels_ref)  # index-0 is the ref view
            if self.aux_loss:
                logit_pred_aux = result["aux"]
                loss_aux = self.criterion(logit_pred_aux, labels_flatten)
                loss += self.weight_aux * loss_aux

            if torch.isnan(loss).item() or torch.isinf(loss).item():
                print(loss.item())

            # abrupt change
            if loss.item() > 1.0 and check_label:
                save_batch_result(os.path.join(self.check_label_dir, "{}".format(step)),
                                  rgb_inputs[:, 0, ...].cpu().numpy(),
                                  labels_ref.cpu().numpy(),
                                  torch.argmax(logit_pred, dim=1).cpu().numpy())

            # Keep track of loss for current epoch
            epoch_loss.append(loss.item())
            # Keep track of the evaluation metric
            self.metric.add(logit_pred.detach(), labels_ref.detach())
            if self.aux_loss:
                self.metric_aux.add(logit_pred_aux.detach(), labels_flatten.detach())

            endTime = get_time()
            avgTime += (endTime - startTime)
            numTimeSteps += 1

            if iteration_loss > 0 and (step % iteration_loss == 0):
                print("[Step: %d/%d (%3.2f ms)] Iteration loss: %.4f" % (step, len(self.data_loader), \
                                                                         1000*(avgTime / (numTimeSteps if numTimeSteps>0 else 1)), loss.item()))
                numTimeSteps = 0
                avgTime = 0.

        ret = {
            "loss": np.nanmean(np.array(epoch_loss)),  # in very rare case, there will be NaNs
            "metric": self.metric.value(),
        }
        if self.aux_loss:
            ret["metric_aux"] = self.metric_aux.value()

        return ret


def save_batch_result(save_dir, rgb_input, label_gt, label_pred):
    """
    :param save_dir:
    :param rgb_input: [B, 3, H, W]
    :param label_gt: [B, H, W]
    :param label_pred: [B, H, W]
    :return:
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    MEAN = PIXEL_MEAN.cpu().numpy()
    STD = PIXEL_STD.cpu().numpy()
    rgb_input = np.transpose(rgb_input, (0, 2, 3, 1))  # [B, H, W, 3]
    B, H, W, _ = rgb_input.shape
    for i in range(B):
        img_to_save = np.zeros((H, W * 3, 3))
        rgb = 255 * (rgb_input[i] * STD + MEAN)
        label_img_gt = create_label_image(label_gt[i], color_encoding_scannet20)
        label_img_pred = create_label_image(label_pred[i], color_encoding_scannet20)
        img_to_save[:, :W, :] = rgb
        img_to_save[:, W:2*W, :] = label_img_gt
        img_to_save[:, 2*W:3*W, :] = label_img_pred
        imageio.imwrite(os.path.join(save_dir, "{}.png".format(i)), img_to_save.astype(np.uint8))


def update_args(cfg, args):    
    for k, v in vars(args).items():
        if v is not None:
            cfg.setdefault(k, v)
    
    cfg.log_dir = args.log_dir
    cfg.event_dir = os.path.join(cfg.log_dir, "events")
    cfg.save_dir = os.path.join(cfg.log_dir, "checkpoints")
    
    return cfg


def get_model(cfg, num_classes=21, pretrained=True, device=torch.device("cuda")):
    model = LatentPriorNetwork(num_classes=num_classes,
                               pretrained=pretrained,
                               modality=cfg.setdefault("modality", "rgbd"),
                               use_ssma=cfg.setdefault("use_ssma", True),
                               reproject=cfg.setdefault("reproject", True),
                               decoder_in_dim=cfg.decoder_in_dim,
                               decoder_feat_dim=cfg.decoder_feat_dim,
                               decoder_head_dim=cfg.decoder_head_dim,
                               window_size=cfg.window_size,
                               projection_dims=cfg.projection_dim,
                               render=cfg.setdefault("render", False),
                               fusion_mode=cfg.fusion_mode).to(device)
    return model


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--scannet_root", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--H", type=int)
    parser.add_argument("--W", type=int)
    parser.add_argument("--depth_err", type=float)
    parser.add_argument("--p_drop_depth", type=float)
    parser.add_argument("--modality", type=str, help="[rgbd, rgb]")
    data_aug_parser = parser.add_mutually_exclusive_group(required=False)
    data_aug_parser.add_argument("--data_aug", dest="data_aug", action="store_true")
    data_aug_parser.add_argument("--no_data_aug", dest="data_aug", action="store_false")
    parser.set_defaults(data_aug=True)
    # parser.add_argument("--use_ssma", dest="use_ssma", action="store_true")
    # parser.add_argument("--reproject", dest="reproject", action="store_true")
    # parser.set_defaults(use_ssma=True, reproject=True)
    parser.add_argument("--skip", type=int)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--fusion_mode", type=str)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--lr_decay", type=float)
    parser.add_argument("--lr_decay_epochs", type=int)
    parser.add_argument("--div_factor", type=float)
    parser.add_argument("--pct_start", type=float)
    parser.add_argument("--final_div_factor", type=float)
    parser.add_argument("--anneal_strategy", type=str)
    parser.add_argument("--weight_aux", type=float)
    parser.add_argument("--decoder_feat_dim", type=int)
    parser.add_argument("--decoder_head_dim", type=int)

    args = parser.parse_args()
    print("Using config file: {}".format(args.config))
    cfg = load_yaml(args.config)
    # update cfg from command line args
    cfg = update_args(cfg, args)
    print(cfg)

    print("Saving logs to: {}".format(cfg.log_dir))
    create_logdir(cfg)
    save_config(cfg, os.path.join(cfg.log_dir, "config.yaml"))
    shutil.copy("networks/LatentPriorNetwork.py", os.path.join(cfg.log_dir, "LatentPriorNetwork.py"))
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=cfg.event_dir)

    # create datasets and data loaders, during training always test with the same resolution as train_set
    data_loaders, class_weights, class_encoding = load_dataset(cfg)
    train_loader, val_loader = data_loaders
    num_classes = len(class_encoding)

    # create model and optimizer
    model = get_model(cfg, device=device)

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
    if cfg.scheduler == "step":
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
    checkpoints = os.listdir(cfg.save_dir)
    if len(checkpoints) > 0:
        chkpt_path = os.path.join(cfg.save_dir, sorted(checkpoints, key=lambda x: int(x[6:-4]))[-1])
        print("Resume training from {}".format(chkpt_path))
        chkpt = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(chkpt["state_dict"])
        optimizer.load_state_dict(chkpt["optimizer"])
        lr_updater.load_state_dict(chkpt["lr_scheduler"])
        start_epoch = int(chkpt["epoch"]) + 1
        start_iter = chkpt["n_iter"] + 1
    else:
        print("Training from scratch...")
        start_epoch = 0
        start_iter = 0

    # Evaluation metric
    ignore_index = list(class_encoding).index('unlabeled')
    metric = IoU(num_classes, ignore_index=ignore_index)

    trainer = LPNTrainer(cfg, model, train_loader, optimizer, criterion, metric, device, lr_scheduler=lr_updater, global_step=start_iter, writer=writer)
    tester = LPNTester(cfg, model, val_loader, criterion_val, metric, device)
    for epoch in range(start_epoch, cfg.epochs):
        # if epoch > 0:
        #     lr_updater.step()
        result_dict = trainer.run_epoch(cfg.print_every)
        epoch_loss, (iou, miou) = result_dict["loss"], result_dict["metric"]
        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))
        writer.add_scalar("Train/epoch_loss", epoch_loss, epoch)
        writer.add_scalar("Train/miou", miou, epoch)
        if cfg.aux_loss:
            iou_aux, miou_aux = result_dict["metric_aux"]
            writer.add_scalar("Train/miou_aux", miou_aux, epoch)

        # validate
        if (epoch + 1) % cfg.validate_every == 0 or epoch + 1 == cfg.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            result_dict = tester.run_epoch(cfg.print_every)
            loss, (iou, miou) = result_dict["loss"], result_dict["metric"]
            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, loss, miou))
            writer.add_scalar("Val/epoch_loss", loss, epoch)
            writer.add_scalar("Val/miou", miou, epoch)
            for key, class_iou in zip(class_encoding.keys(), iou):
                print("{0}: {1:.4f}".format(key, class_iou))

            if cfg.aux_loss:
                iou_aux, miou_aux = result_dict["metric_aux"]
                writer.add_scalar("Val/miou_aux", miou_aux, epoch)

        # save checkpoint
        if epoch + 1 == cfg.epochs or (epoch + 1) % cfg.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'n_iter': trainer.global_step,
                'miou': miou,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_updater.state_dict(),
            }
            torch.save(checkpoint, os.path.join(cfg.save_dir, "chkpt-{}.pth".format(epoch)))


if __name__ == "__main__":
    train()
