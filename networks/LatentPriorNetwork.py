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

import random
import time

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from networks.MobileNet import mobilenet_v3_large, ConvNormActivation
from networks.conv_module import depthwise_separable_conv
from networks.rend_utils import warp_feature_batch, warp_feature_batch_render


# TODO: shouldn't be put here...
def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
                          bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels,
                      out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class SSMA(nn.Module):
    def __init__(self, input_channels, compression_rate=6):
        super(SSMA, self).__init__()
        self.scale_layer = nn.Sequential(nn.Conv2d(2 * input_channels, input_channels // compression_rate, 3, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(
                                             input_channels // compression_rate, 2 * input_channels, 3, padding=1),
                                         nn.Sigmoid())
        self.final_layer = nn.Sequential(nn.Conv2d(2 * input_channels, input_channels, 3, padding=1),
                                         nn.BatchNorm2d(input_channels))

    def forward(self, rgb_feat, depth_feat):
        feat_cat = torch.cat([rgb_feat, depth_feat], dim=1)  # [B, 2C, H, W]
        scale_fc = self.scale_layer(feat_cat)  # [B, 2C, H, W]
        return self.final_layer(feat_cat * scale_fc)


class FusionFeatures(nn.Module):
    def __init__(self, mode="average"):
        super(FusionFeatures, self).__init__()
        assert mode in ["average", "max"]
        self.mode = mode

    def forward(self, mv_features):
        """
        :param mv_features: N * [B, C, H, W], index=0 always corresponds to ref-view
        :return: fused features [B, C, H, W]
        """
        if self.mode == "average":
            feats_ref = mv_features[0].unsqueeze(1)  # [B, 1, C, H, W]
            B, _, _, H, W = feats_ref.shape
            feats_warped = torch.stack(
                mv_features[1:], dim=1)  # [B, N-1， C， H， W]
            visible_mask = (feats_warped.abs() > 0.).any(
                dim=2)  # [B, N-1, H, W]
            # [B, N, 1, H, W]
            visible_weight = torch.cat([torch.ones(B, 1, H, W).to(
                feats_ref), visible_mask.float()], dim=1).unsqueeze(2)
            # fuse [B, C, H, W]
            feats_fused = (torch.cat(
                [feats_ref, feats_warped], dim=1) * visible_weight).sum(1) / visible_weight.sum(1)
        elif self.mode == "max":
            feats = torch.stack(mv_features, dim=1)  # [B, N, C, H, W]
            feats_fused, _ = torch.max(feats, dim=1)
        else:
            raise NotImplementedError

        return feats_fused


# 2D Semantic Segmentation Model
class LatentPriorNetwork(nn.Module):
    """
    This is our LatentPriorNetwork (LPN) that supports:
    1. Multi-view RGBD (default): rgb and depth fusion with SSMA + feature warping (w/ depth, camera poses and K)
    2. Multi-view RGBD with RGB feature: rgb-only encoder, no SSMA, depth only used in feature warping (w/ depth, camera poses and K)
    3. Single-view RGBD: rgb and depth fusion with SSMA
    4. Single-view RGB: rgb input only

    -----------------------------------------------------------------------------
       |  use_depth  use_ssma   reproject
    -----------------------------------------------------------------------------
    1. |      T          T          T (feat-warp)
    2. |      T          F          T (feat-warp)
    3. |      T          T          F
    4. |      F          F          F
    -----------------------------------------------------------------------------
    """

    def __init__(self, num_classes=21, pretrained=True, modality="rgbd", reproject=True, use_ssma=True,
                 decoder_in_dim=256, decoder_feat_dim=256, decoder_head_dim=128, window_size=3,
                 projection_dims=[128, 32, 64], render=False, fusion_mode="concat"):
        super(LatentPriorNetwork, self).__init__()
        assert modality in ["rgb", "rgbd"], "Unknown modality type!!!"
        self.modality = modality  # rgbd or rgb
        # use_depth == True means that depth is required either for feature extraction OR feature reprojection
        self.use_depth = (self.modality == "rgbd")
        self.use_ssma = use_ssma  # use_ssma == True means feature is extracted from RGB+depth
        self.reproject = reproject  # whether to perform feature extraction or not
        # window_size > 1 doesn't make sense for reproject == False

        # original deeplab model
        self.rgb_encoder = mobilenet_v3_large(
            pretrained=pretrained, input_channels=3, dilated=True)
        if self.use_ssma:
            self.depth_encoder = mobilenet_v3_large(
                pretrained=pretrained, input_channels=1, dilated=True)
            # fuse (RGB and D) output of stage 1, 2, 5, stage 1 and 2 will also be used in skip connection
            self.fusion_layer1 = SSMA(
                self.rgb_encoder.stage1[-1].out_channels, compression_rate=6)
            self.fusion_layer2 = SSMA(
                self.rgb_encoder.stage2[-1].out_channels, compression_rate=6)
            self.fusion_layer3 = SSMA(
                self.rgb_encoder.stage4[-1].out_channels, compression_rate=6)
        bneck_feat_dim = self.rgb_encoder.lastconv.out_channels
        self.ASPP = ASPP(bneck_feat_dim, [12, 24, 36])

        # projection layers for features: feat, skip1, skip2, before warping
        self.project_feat = nn.Sequential(nn.Conv2d(256, projection_dims[0], 1, bias=False),
                                          nn.BatchNorm2d(projection_dims[0]),
                                          nn.ReLU())
        # skip1: 24, 1/4
        self.project_skip1 = nn.Sequential(nn.Conv2d(24, projection_dims[1], 1, bias=False),
                                           nn.BatchNorm2d(projection_dims[1]),
                                           nn.ReLU())
        # skip2: 40, 1/8
        self.project_skip2 = nn.Sequential(nn.Conv2d(40, projection_dims[2], 1, bias=False),
                                           nn.BatchNorm2d(projection_dims[2]),
                                           nn.ReLU())

        # bottleneck, feature re-projection goes in here:
        self.window_size = window_size
        self.muti_view = window_size > 1

        # use py3d rendering or simple warping, only works when multi_view==True
        self.render = render
        # print("Using py3d render: {}".format(self.render))

        # Fuse features from different views size
        assert fusion_mode in ["concat", "average", "max"], "Unknown feature fusion mode"
        self.fusion_mode = fusion_mode
        if self.fusion_mode == "concat":
            self.fuse_feat = nn.Sequential(depthwise_separable_conv(projection_dims[0] * window_size, decoder_in_dim, 3, padding=1),
                                           depthwise_separable_conv(decoder_in_dim, decoder_in_dim, 3, padding=1))  # 1/16
            self.fuse_skip1 = nn.Sequential(depthwise_separable_conv(projection_dims[1] * window_size, 32, 3, padding=1),
                                            depthwise_separable_conv(32, 32, 3, padding=1))  # 1/4
            self.fuse_skip2 = nn.Sequential(depthwise_separable_conv(projection_dims[2] * window_size, 64, 3, padding=1),
                                            depthwise_separable_conv(64, 64, 3, padding=1))  # 1/8
        else:
            self.fuse_feat = FusionFeatures(mode=self.fusion_mode)
            self.fuse_skip1 = FusionFeatures(mode=self.fusion_mode)
            self.fuse_skip2 = FusionFeatures(mode=self.fusion_mode)

        self.render_radius = {120: 0.03, 60: 0.05, 30: 0.10, 15: 0.20}

        # UNet type decoder with skip connections
        self.decoder = SemanticDecoder(decoder_in_dim,
                                       decoder_channels=decoder_feat_dim,
                                       head_channels=decoder_head_dim,
                                       skip_dims=projection_dims[1:],
                                       num_classes=num_classes)

    def feature_net_forward(self, rgb_input, depth_input=None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Extract features from only single view
        :param rgb_input: [B, 3, H, W]
        :param depth_input: [B, 1, H, W] or None
        :return:
        """

        B, _, H, W = rgb_input.shape  # [H, W]
        # If use_ssma==True but depth_input is None, set depth input to all zeros
        if depth_input is None and self.use_ssma:
            depth_input = torch.zeros(B, 1, H, W).to(rgb_input)

        # stage 0: 16, 1/2
        rgb0 = self.rgb_encoder.forward_first_conv(rgb_input)
        depth0 = self.depth_encoder.forward_first_conv(
            depth_input) if depth_input is not None else None
        # stage 1: 24, 1/4
        rgb1 = self.rgb_encoder.forward_stage1(rgb0)
        depth1 = self.depth_encoder.forward_stage1(
            depth0) if depth0 is not None else None
        fuse1 = self.fusion_layer1(
            rgb1, depth1) if self.use_ssma else rgb1  # 24, 1/4, no ReLU
        # stage 2: 40, 1/8
        rgb2 = self.rgb_encoder.forward_stage2(fuse1)
        depth2 = self.depth_encoder.forward_stage2(
            depth1) if depth1 is not None else None
        fuse2 = self.fusion_layer2(
            rgb2, depth2) if self.use_ssma else rgb2  # 40, 1/8, no ReLU
        # stage 3: 80, 1/16
        rgb3 = self.rgb_encoder.forward_stage3(fuse2)
        depth3 = self.depth_encoder.forward_stage3(
            depth2) if depth2 is not None else None
        # stage 4: 160, 1/16
        rgb4 = self.rgb_encoder.forward_stage4(rgb3)
        depth4 = self.depth_encoder.forward_stage4(
            depth3) if depth3 is not None else None
        fuse3 = self.fusion_layer3(
            rgb4, depth4) if self.use_ssma else rgb4  # 160, 1/16
        # stage 5: 160, 1/16
        rgb5 = self.rgb_encoder.forward_stage5(fuse3)
        # depth5 = self.depth_encoder.forward_stage5(depth4)

        # early feature
        early_features = self.rgb_encoder.lastconv(rgb5)  # 960, H/16, W/16
        # late feature
        late_features = self.ASPP(early_features)  # 256, H/16, W/16 after ReLU

        # Feature maps used in decoder
        # bneck feature
        # 128, H/16, W/16, after ReLU
        bneck_feat = self.project_feat(late_features)
        skip1 = self.project_skip1(fuse1)  # 32, H/4, W/4, after ReLU
        skip2 = self.project_skip2(fuse2)  # 64, H/8, W/8, after ReLU

        return skip1, skip2, bneck_feat

    def single_view_forward(self, rgb_input, depth_input=None) -> Dict[str, Tensor]:
        """
        single_view forward pass
        :param rgb_input: [B, 3, H, W]
        :param depth_input: [B, 1, H, W] or None
        :return:
        """

        skip1, skip2, feat = self.feature_net_forward(
            rgb_input, depth_input=depth_input)
        label, last_feat = self.decoder(feat, skip1, skip2)
        ret = {
            "out": label,
            "last_feat": last_feat,
            # intermediate feats used for re-projection
            "skip1": skip1,
            "skip2": skip2,
            "feat": feat
        }
        return ret

    def forward(self, rgb_input, depth_input, K_list, c2w_list, w2c_list, aux_loss=False, p_drop_depth=0.0) -> Dict[str, Tensor]:
        """
        multi_view forward pass (for training OR batched inference)
        :param rgb_input: [B, N, 3, H, W]
        :param depth_input: [B, N, 1, H, W]
        :param K_list: [B, 3, 3]
        :param c2w_list: [B, N, 4, 4]
        :param w2c_list: [B, N, 4, 4]
        :param aux_loss: whether to predict aux labels, i.e. single-view output
        :return:
        """

        B, N, _, H, W = rgb_input.shape  # [H, W]

        drop_depth_flag = False
        if not self.use_ssma or not self.use_depth:
            drop_depth_flag = True
        if p_drop_depth > 0.0 and random.random() < p_drop_depth:
            drop_depth_flag = True

        # 1/4, 1/8, 1/16
        # [B, 3, H, W]
        ref_skip1, ref_skip2, ref_feat = self.feature_net_forward(rgb_input[:, 0, ...],
                                                                  depth_input=None if drop_depth_flag else depth_input[:, 0, ...])
        feat, skip1, skip2 = [ref_feat], [ref_skip1], [ref_skip2]

        # depth_ref = depth_input[:, 0, ...]
        # c2w_ref = c2w_list[:, 0, ...]
        if aux_loss:
            feat_aux, skip1_aux, skip2_aux = [
                ref_feat], [ref_skip1], [ref_skip2]

        for n in range(1, self.window_size):
            # get multi-level features of the other views
            src_skip1, src_skip2, src_feat = self.feature_net_forward(rgb_input[:, n, ...],
                                                                      depth_input=None if drop_depth_flag else depth_input[:, n, ...])
            # for auxiliary output
            if aux_loss:
                feat_aux.append(src_feat)
                skip1_aux.append(src_skip1)
                skip2_aux.append(src_skip2)

            # get warped features
            K1 = K_list.clone()
            K1[:, 0:2, :] /= 4.
            if self.render:
                depth_scaled = F.interpolate(
                    depth_input[:, n, ...], scale_factor=1/4., mode="nearest")
                warped_skip1 = warp_feature_batch_render(src_skip1, depth_scaled,
                                                         K1, w2c_list[:,
                                                                      0, ...],
                                                         c2w_list[:, n, ...], K1, radius=self.render_radius[H // 4])
            else:
                depth_scaled = F.interpolate(
                    depth_input[:, 0, ...], scale_factor=1/4., mode="nearest")
                warped_skip1 = warp_feature_batch(src_skip1, depth_scaled,
                                                  K1, c2w_list[:, 0, ...], w2c_list[:, n, ...], K1)
            skip1.append(warped_skip1)
            K2 = K_list.clone()
            K2[:, 0:2, :] /= 8.
            if self.render:
                depth_scaled = F.interpolate(
                    depth_input[:, n, ...], scale_factor=1/8., mode="nearest")
                warped_skip2 = warp_feature_batch_render(src_skip2, depth_scaled,
                                                         K2, w2c_list[:,
                                                                      0, ...],
                                                         c2w_list[:, n, ...], K2, radius=self.render_radius[H // 8])
            else:
                depth_scaled = F.interpolate(
                    depth_input[:, 0, ...], scale_factor=1/8., mode="nearest")
                warped_skip2 = warp_feature_batch(src_skip2, depth_scaled,
                                                  K2, c2w_list[:, 0, ...], w2c_list[:, n, ...], K2)
            skip2.append(warped_skip2)
            K3 = K_list.clone()
            K3[:, 0:2, :] /= 16.
            if self.render:
                depth_scaled = F.interpolate(
                    depth_input[:, n, ...], scale_factor=1/16., mode="nearest")
                warped_feat = warp_feature_batch_render(src_feat, depth_scaled,
                                                        K3, w2c_list[:,
                                                                     0, ...],
                                                        c2w_list[:, n, ...], K3, radius=self.render_radius[H // 16])
            else:
                depth_scaled = F.interpolate(
                    depth_input[:, 0, ...], scale_factor=1/16., mode="nearest")
                warped_feat = warp_feature_batch(src_feat, depth_scaled,
                                                 K3, c2w_list[:, 0, ...], w2c_list[:, n, ...], K3)
            feat.append(warped_feat)

        # concat and project
        if self.fusion_mode == "concat":
            # N * [B, C, H/16, W/16] -> [B, C, H/16, W/16]
            fused_feat = self.fuse_feat(torch.cat(feat, dim=1))
            # N * [B, C, H/4, W/4] -> [B, C, H/4, W/4]
            fused_skip1 = self.fuse_skip1(torch.cat(skip1, dim=1))
            # N * [B, C, H/8, W/8] -> [B, C, H/8, W/8]
            fused_skip2 = self.fuse_skip2(torch.cat(skip2, dim=1))
        else:
            # N * [B, C, H/16, W/16] -> [B, C, H/16, W/16]
            fused_feat = self.fuse_feat(feat)
            # N * [B, C, H/4, W/4] -> [B, C, H/4, W/4]
            fused_skip1 = self.fuse_skip1(skip1)
            # N * [B, C, H/8, W/8] -> [B, C, H/8, W/8]
            fused_skip2 = self.fuse_skip2(skip2)

        result = OrderedDict()
        label, last_feat = self.decoder(fused_feat, fused_skip1, fused_skip2)
        result["out"] = label
        result["feat"] = last_feat

        if aux_loss:
            # must first stack along dim1 and then squeeze dim0 and dim1
            feat_aux = torch.stack(feat_aux, dim=1).view(
                B * N, -1, H // 16, W // 16)  # [BN, C, H/16, W/16]
            skip1_aux = torch.stack(
                skip1_aux, dim=-1).view(B * N, -1, H // 4, W // 4)  # [BN, C, H/4, W/4]
            skip2_aux = torch.stack(skip2_aux, dim=1).view(
                B * N, -1, H // 8, W // 8)  # [BN, C, H/8, W/8]
            aux_label, _ = self.decoder(feat_aux, skip1_aux, skip2_aux)
            result["aux"] = aux_label

        return result

    def causal_forward(self, mem_bank, K, profiler=None):
        """
        :param mem_bank: queue of tuples, each tuple represents a frame (skip1, skip2, feat, c2w, w2c, frame_id)
        :param depth_ref: [1, 1, H, W]
        :return:
        """
        n_views = len(mem_bank)
        ref_skip1, ref_skip2, ref_feat, depth_ref, c2w_ref, w2c_ref, _ = mem_bank[-1]
        if depth_ref is not None:
            H = depth_ref.shape[2]
        if n_views == 1:
            t1 = get_time()
            label, _ = self.decoder(ref_feat, ref_skip1, ref_skip2)
            t2 = get_time()
            if profiler is not None:
                profiler.append_decoder(t2 - t1)
            ret_dict = {
                "out": label
            }
        else:
            skip1, skip2, feat = [ref_skip1], [ref_skip2], [ref_feat]
            if profiler is not None:
                warping_time1, warping_time2, warping_time3 = [], [], []
            for n in range(n_views - 1):
                start = get_time()
                src_skip1, src_skip2, src_feat, depth_src, c2w_src, w2c_src, _ = mem_bank[n]

                K1 = K.clone()
                K1[:, 0:2, :] /= 4.

                if self.render:
                    depth_scaled = F.interpolate(
                        depth_src, scale_factor=1/4., mode="nearest")
                    warped_skip1 = warp_feature_batch_render(
                        src_skip1, depth_scaled, K1, w2c_ref, c2w_src, K1, radius=self.render_radius[H // 4])
                else:
                    depth_scaled = F.interpolate(
                        depth_ref, scale_factor=1/4., mode="nearest")
                    warped_skip1 = warp_feature_batch(
                        src_skip1, depth_scaled, K1, c2w_ref, w2c_src, K1)
                skip1.append(warped_skip1)
                t1 = get_time()

                K2 = K.clone()
                K2[:, 0:2, :] /= 8.
                if self.render:
                    depth_scaled = F.interpolate(
                        depth_src, scale_factor=1/8., mode="nearest")
                    warped_skip2 = warp_feature_batch_render(
                        src_skip2, depth_scaled, K2, w2c_ref, c2w_src, K2, radius=self.render_radius[H // 8])
                else:
                    depth_scaled = F.interpolate(
                        depth_ref, scale_factor=1/8., mode="nearest")
                    warped_skip2 = warp_feature_batch(
                        src_skip2, depth_scaled, K2, c2w_ref, w2c_src, K2)
                skip2.append(warped_skip2)
                t2 = get_time()

                K3 = K.clone()
                K3[:, 0:2, :] /= 16.
                if self.render:
                    depth_scaled = F.interpolate(
                        depth_src, scale_factor=1/16., mode="nearest")
                    warped_feat = warp_feature_batch_render(
                        src_feat, depth_scaled, K3, w2c_ref, c2w_src, K3, radius=self.render_radius[H // 16])
                else:
                    depth_scaled = F.interpolate(
                        depth_ref, scale_factor=1/16., mode="nearest")
                    warped_feat = warp_feature_batch(
                        src_feat, depth_scaled, K3, c2w_ref, w2c_src, K3)
                feat.append(warped_feat)

                end = get_time()
                if profiler is not None:
                    warping_time1.append(t1 - start)
                    warping_time2.append(t2 - t1)
                    warping_time3.append(end - t2)

            if profiler is not None:
                profiler.append_feature_skip1(np.asarray(warping_time1).mean())
                profiler.append_feature_skip2(np.asarray(warping_time2).mean())
                profiler.append_feature_bneck(np.asarray(warping_time3).mean())

            start = get_time()
            if self.fusion_mode == "concat":
                # N * [B, C, H/16, W/16] -> [B, C, H/16, W/16]
                fused_feat = self.fuse_feat(torch.cat(feat, dim=1))
                # N * [B, C, H/4, W/4] -> [B, C, H/4, W/4]
                fused_skip1 = self.fuse_skip1(torch.cat(skip1, dim=1))
                # N * [B, C, H/8, W/8] -> [B, C, H/8, W/8]
                fused_skip2 = self.fuse_skip2(torch.cat(skip2, dim=1))
            else:
                # N * [B, C, H/16, W/16] -> [B, C, H/16, W/16]
                fused_feat = self.fuse_feat(feat)
                # N * [B, C, H/4, W/4] -> [B, C, H/4, W/4]
                fused_skip1 = self.fuse_skip1(skip1)
                # N * [B, C, H/8, W/8] -> [B, C, H/8, W/8]
                fused_skip2 = self.fuse_skip2(skip2)
            end = get_time()
            if profiler is not None:
                profiler.append_feature_fusion(end - start)

            start = get_time()
            label, _ = self.decoder(fused_feat, fused_skip1, fused_skip2)
            end = get_time()
            if profiler is not None:
                profiler.append_decoder(end - start)

            ret_dict = {
                "out": label,
                "skip1": fused_skip1,
                "skip2": fused_skip2,
                "feat": fused_feat
            }
        return ret_dict

    def causal_forward_batch_warp(self, mem_bank, K, profiler=None):
        """
        :param mem_bank: queue of tuples, each tuple represents a frame (skip1, skip2, feat, c2w, w2c, frame_id)
        :param depth_ref: [1, 1, H, W]
        :return:
        """
        n_views = len(mem_bank)
        ref_skip1, ref_skip2, ref_feat, depth_ref, c2w_ref, w2c_ref, _ = mem_bank[-1]
        H = depth_ref.shape[2]
        if n_views == 1:
            t1 = get_time()
            label, _ = self.decoder(ref_feat, ref_skip1, ref_skip2)
            t2 = get_time()
            if profiler is not None:
                profiler.append_decoder(t2 - t1)
            ret_dict = {
                "out": label
            }
        else:
            skip1, skip2, feat = [ref_skip1], [ref_skip2], [ref_feat]
            if profiler is not None:
                warping_time1, warping_time2, warping_time3 = [], [], []
            start = get_time()
            skip1_src, skip2_src, feat_src, depth_src, c2w_src, w2c_src = [], [], [], [], [], []
            for n in range(n_views - 1):
                skip1_, skip2_, feat_, depth_, c2w_, w2c_, _ = mem_bank[n]
                skip1_src.append(skip1_)
                skip2_src.append(skip2_)
                feat_src.append(feat_)
                depth_src.append(depth_)
                c2w_src.append(c2w_)
                w2c_src.append(w2c_)
            skip1_src = torch.cat(skip1_src, dim=0)  # [N, C, H, W]
            skip2_src = torch.cat(skip2_src, dim=0)
            feat_src = torch.cat(feat_src, dim=0)
            depth_src = torch.cat(depth_src, dim=0)
            c2w_src = torch.cat(c2w_src, dim=0)
            w2c_src = torch.cat(w2c_src, dim=0)
            w2c_ref = w2c_ref.repeat(n_views - 1, 1, 1)
            c2w_ref = c2w_ref.repeat(n_views - 1, 1, 1)

            # remove this big for-loop
            # [1, 3, 3]
            K1 = K.clone()
            K1[:, 0:2, :] /= 4.
            K1 = K1.repeat(n_views - 1, 1, 1)
            if self.render:
                depth_scaled = F.interpolate(
                    depth_src, scale_factor=1/4., mode="nearest")
                warped_skip1 = warp_feature_batch_render(
                    skip1_src, depth_scaled, K1, w2c_ref, c2w_src, K1, radius=self.render_radius[H // 4])
            else:
                depth_scaled = F.interpolate(
                    depth_ref, scale_factor=1/4., mode="nearest")
                warped_skip1 = warp_feature_batch(
                    skip1_src, depth_scaled, K1, c2w_ref, w2c_src, K1)
            warped_skip1 = list(torch.unbind(warped_skip1.unsqueeze(0), dim=1))
            skip1 += warped_skip1
            t1 = get_time()

            K2 = K.clone()
            K2[:, 0:2, :] /= 8.
            K2 = K2.repeat(n_views - 1, 1, 1)
            if self.render:
                depth_scaled = F.interpolate(
                    depth_src, scale_factor=1/8., mode="nearest")
                warped_skip2 = warp_feature_batch_render(
                    skip2_src, depth_scaled, K2, w2c_ref, c2w_src, K2, radius=self.render_radius[H // 8])
            else:
                depth_scaled = F.interpolate(
                    depth_ref, scale_factor=1/8., mode="nearest")
                warped_skip2 = warp_feature_batch(
                    skip2_src, depth_scaled, K2, c2w_ref, w2c_src, K2)
            warped_skip2 = list(torch.unbind(warped_skip2.unsqueeze(0), dim=1))
            skip2 += warped_skip2
            t2 = get_time()

            K3 = K.clone()
            K3[:, 0:2, :] /= 16.
            K3 = K3.repeat(n_views - 1, 1, 1)
            if self.render:
                depth_scaled = F.interpolate(
                    depth_src, scale_factor=1/16., mode="nearest")
                warped_feat = warp_feature_batch_render(
                    feat_src, depth_scaled, K3, w2c_ref, c2w_src, K3, radius=self.render_radius[H // 16])
            else:
                depth_scaled = F.interpolate(
                    depth_ref, scale_factor=1/16., mode="nearest")
                warped_feat = warp_feature_batch(
                    feat_src, depth_scaled, K3, c2w_ref, w2c_src, K3)
            warped_feat = list(torch.unbind(warped_feat.unsqueeze(0), dim=1))
            feat += warped_feat

            end = get_time()
            if profiler is not None:
                warping_time1.append(t1 - start)
                warping_time2.append(t2 - t1)
                warping_time3.append(end - t2)

            if profiler is not None:
                profiler.append_feature_skip1(np.asarray(warping_time1).mean())
                profiler.append_feature_skip2(np.asarray(warping_time2).mean())
                profiler.append_feature_bneck(np.asarray(warping_time3).mean())

            start = get_time()
            if self.fusion_mode == "concat":
                # N * [B, C, H/16, W/16] -> [B, C, H/16, W/16]
                fused_feat = self.fuse_feat(torch.cat(feat, dim=1))
                # N * [B, C, H/4, W/4] -> [B, C, H/4, W/4]
                fused_skip1 = self.fuse_skip1(torch.cat(skip1, dim=1))
                # N * [B, C, H/8, W/8] -> [B, C, H/8, W/8]
                fused_skip2 = self.fuse_skip2(torch.cat(skip2, dim=1))
            else:
                # N * [B, C, H/16, W/16] -> [B, C, H/16, W/16]
                fused_feat = self.fuse_feat(feat)
                # N * [B, C, H/4, W/4] -> [B, C, H/4, W/4]
                fused_skip1 = self.fuse_skip1(skip1)
                # N * [B, C, H/8, W/8] -> [B, C, H/8, W/8]
                fused_skip2 = self.fuse_skip2(skip2)
            end = get_time()
            if profiler is not None:
                profiler.append_feature_fusion(end - start)

            start = get_time()
            label, _ = self.decoder(fused_feat, fused_skip1, fused_skip2)
            end = get_time()
            if profiler is not None:
                profiler.append_decoder(end - start)

            ret_dict = {
                "out": label,
                "skip1": fused_skip1,
                "skip2": fused_skip2,
                "feat": fused_feat
            }

        return ret_dict


class SemanticDecoder(nn.Module):
    """
    Decoder follows Panoptic-DeepLab architecture
    """

    def __init__(
            self,
            input_channels,  # output channel from ASPP
            decoder_channels=256,
            head_channels=256,
            skip_dims=[32, 64],
            num_classes=21
    ):
        super(SemanticDecoder, self).__init__()

        self.upsample = torch.nn.Upsample(scale_factor=2,
                                          mode='bilinear',
                                          align_corners=False)

        # upsample x2 -> 256, 1/8

        # concat with skip2 40  stride = 1/8
        self.conv1 = depthwise_separable_conv(
            input_channels + skip_dims[1], decoder_channels, 5, padding=2)

        # concat with skip1 24  stride = 1/4
        self.conv2 = depthwise_separable_conv(
            decoder_channels + skip_dims[0], decoder_channels, 5, padding=2)

        # output head
        self.semantic_head = nn.Sequential(
            depthwise_separable_conv(
                decoder_channels, head_channels, 5, padding=2),
            nn.Conv2d(head_channels, num_classes, 1)
        )
        # Followed by 2x upsample x2

        # weight initialization, seems unnecessary
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out")
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.zeros_(m.bias)

    def forward(self, feat, skip1, skip2):
        # 256, 1/8
        conv0 = self.upsample(feat)
        conv0 = torch.cat([conv0, skip2], dim=1)
        # 256, 1/4
        conv1 = self.upsample(self.conv1(conv0))
        conv1 = torch.cat([conv1, skip1], dim=1)
        conv2 = self.conv2(conv1)
        # output: num_classes, 1
        last_feat = self.semantic_head[0](conv2)
        out_sem = self.semantic_head[1](last_feat)
        out_sem = self.upsample(self.upsample(out_sem))
        return out_sem, last_feat


# Panoptic Segmantation Model
class PanopticRGBD(nn.Module):
    def __init__(self, num_classes=21, pretrained=True, upsample_mode="conv", multi_res_loss=False):
        super(PanopticRGBD, self).__init__()
        # original deeplab model
        self.rgb_encoder = mobilenet_v3_large(
            pretrained=pretrained, input_channels=3, dilated=True)
        self.depth_encoder = mobilenet_v3_large(
            pretrained=pretrained, input_channels=1, dilated=True)
        bneck_feat_dim = self.rgb_encoder.lastconv.out_channels
        self.ASPP_sem = ASPP(bneck_feat_dim, [12, 24, 36])
        self.ASPP_ins = ASPP(bneck_feat_dim, [12, 24, 36])

        # TODO: put some settings into config
        # UNet type decoder with skip connections
        # self.decoder = PanopticDecoder(256, num_classes=num_classes,
        #                                n_layers=2, expand=2, emb_dim=32,
        #                                upsample_mode=upsample_mode)
        self.decoder = PanopticDecoderThreeBranch(256, num_classes=num_classes,
                                                  n_layers=2, expand=2, emb_dim=32,
                                                  upsample_mode=upsample_mode)

        # fuse output of stage 1, 3, 5, also used as skip connection
        self.fusion_layer1 = SSMA(
            self.rgb_encoder.stage1[-1].out_channels, compression_rate=6)
        self.fusion_layer2 = SSMA(
            self.rgb_encoder.stage3[-1].out_channels, compression_rate=6)
        self.fusion_layer3 = SSMA(
            self.rgb_encoder.stage5[-1].out_channels, compression_rate=6)

    def forward(self, rgb_input: Tensor, depth_input: Tensor) -> Dict[str, Tensor]:
        input_shape = rgb_input.shape[-2:]  # [H, W]
        rgb0 = self.rgb_encoder.forward_first_conv(rgb_input)
        depth0 = self.depth_encoder.forward_first_conv(depth_input)
        # stage 1
        rgb1 = self.rgb_encoder.forward_stage1(rgb0)
        depth1 = self.depth_encoder.forward_stage1(depth0)
        fuse1 = self.fusion_layer1(rgb1, depth1)
        # stage 2
        rgb2 = self.rgb_encoder.forward_stage2(fuse1)
        depth2 = self.depth_encoder.forward_stage2(depth1)
        # stage 3
        rgb3 = self.rgb_encoder.forward_stage3(rgb2)
        depth3 = self.depth_encoder.forward_stage3(depth2)
        fuse2 = self.fusion_layer2(rgb3, depth3)
        # stage 4
        rgb4 = self.rgb_encoder.forward_stage4(fuse2)
        depth4 = self.depth_encoder.forward_stage4(depth3)
        # stage 5
        rgb5 = self.rgb_encoder.forward_stage5(rgb4)
        depth5 = self.depth_encoder.forward_stage5(depth4)
        fuse3 = self.fusion_layer3(rgb5, depth5)
        # early feature
        early_features = self.rgb_encoder.lastconv(fuse3)  # 960, H/16, W/16

        # late feature after ASPP
        sem_late_features = self.ASPP_sem(early_features)  # 256 after ReLU
        ins_late_features = self.ASPP_ins(early_features)  # 256 after ReLU

        # get predictions
        result = OrderedDict()
        out_sem, out_emb, out_cen = self.decoder(
            sem_late_features, ins_late_features, fuse1, fuse2)
        result["out_sem"] = out_sem
        result["out_emb"] = out_emb
        result["out_cen"] = out_cen

        return result


class PanopticDecoder(nn.Module):
    def __init__(
            self,
            input_channels,
            n_layers=2,
            expand=2,
            num_classes=21,
            emb_dim=32,
            upsample_mode="conv",
    ):
        super(PanopticDecoder, self).__init__()

        assert upsample_mode in ["conv", "bilinear"]
        self.upsample_mode = upsample_mode

        self.upsample = torch.nn.Upsample(scale_factor=2,
                                          mode='bilinear',
                                          align_corners=False)

        """ Semantic Decoder """
        # first conv layer
        self.sem_firstconv = nn.Sequential(
            ConvNormActivation(input_channels, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU),
            ConvNormActivation(80, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=None)
        )
        # concat with skip2 80  stride = 1/16
        self.sem_conv1 = create_decoder_block_conv_ds(
            160, 80, expand=expand, n_layers=n_layers)
        # Followed by an up-sampling x2
        # stride = 1/8
        self.sem_conv2 = create_decoder_block_conv_ds(
            80, 40, expand=expand, n_layers=n_layers)
        # Followed by an up-sampling x2
        # concat with skip1 24  stride = 1/4
        self.sem_conv3 = create_decoder_block_conv_ds(
            64, 64, expand=expand, n_layers=n_layers)
        # Followed by up-sampling x2  sride = 1/2
        self.sem_lastconv = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(),
                                          nn.Conv2d(64, num_classes, 1))
        # Followed by up-sampling x2 to original resolution

        """ Instance Decoder"""
        # shared part
        self.ins_firstconv = nn.Sequential(
            ConvNormActivation(input_channels, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU),
            ConvNormActivation(80, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=None)
        )
        # concat with skip2 80  stride = 1/16
        # self.ins_conv1 = create_decoder_block_conv_ds(160, 80, expand=expand, n_layers=n_layers)
        self.ins_conv1 = ObjDecoderModule(160, 128, nr_decoder_blocks=n_layers)
        # Followed by an up-sampling x2
        # stride = 1/8
        # self.ins_conv2 = create_decoder_block_conv_ds(80, 40, expand=expand, n_layers=n_layers)
        self.ins_conv2 = ObjDecoderModule(128, 104, nr_decoder_blocks=n_layers)
        # concat with skip1 24  stride = 1/4

        # self.ins_conv3 = create_decoder_block_conv_ds(64, emb_dim, expand=expand, n_layers=n_layers)
        # self.ins_conv3 = ObjDecoderModule(64, emb_dim, nr_decoder_blocks=n_layers)
        # ins-decoder final shared feature

        # embedding head
        # self.ins_emb_up1 = partial(F.interpolate, mode="bilinear", align_corners=False)
        # self.ins_emb_conv1 = create_decoder_block_conv_ds(emb_dim, emb_dim, expand=expand, n_layers=n_layers)
        self.ins_emb_conv1 = ObjDecoderModule(
            128, emb_dim, nr_decoder_blocks=n_layers)
        # 1/2
        self.ins_emb_conv2 = ObjDecoderModule(
            emb_dim, emb_dim, nr_decoder_blocks=n_layers)
        # 1
        self.ins_emb_lastconv = nn.Sequential(nn.Conv2d(emb_dim, emb_dim, 3, padding=1),
                                              nn.BatchNorm2d(emb_dim),
                                              nn.ReLU(),
                                              nn.Conv2d(emb_dim, emb_dim, 3, padding=1))

        # center head
        # self.ins_cen_up1 = partial(F.interpolate, mode="bilinear", align_corners=False)
        # self.ins_cen_conv1 = create_decoder_block_conv_ds(emb_dim, 8, expand=expand, n_layers=n_layers)
        self.ins_cen_conv1 = ObjDecoderModule(
            128, 128, nr_decoder_blocks=n_layers)
        # 1/2
        # self.ins_cen_conv2 = ObjDecoderModule(64, 1, nr_decoder_blocks=n_layers)
        # self.ins_cen_conv1 = nn.Sequential(nn.Conv2d(emb_dim, 8, 3), nn.BatchNorm2d(8), nn.ReLU(),
        #                                    nn.Conv2d(8, 8, 3), nn.BatchNorm2d(8), nn.ReLU())
        self.ins_cen_lastconv = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            # nn.BatchNorm2d(emb_dim),
            # nn.ReLU(),
            # nn.Conv2d(emb_dim, 1, 1)
        )

        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        #         if m.out_channels == num_classes or m.out_channels == 1 or m.groups == m.in_channels:
        #             continue
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        # print("initialized decoder!")

    def forward(self, sem_feat, ins_feat, skip1, skip2):
        """
        :param sem_feat: feature output from semantic-ASPP
        :param ins_feat: feature output from inatance-ASPP
        :param skip1: feat skip1 C=80
        :param skip2: feat skip2 C=24
        :return:
        """

        sem_conv0 = self.sem_firstconv(sem_feat)  # 1/16
        sem_conv0 = torch.cat([sem_conv0, skip2], dim=1)
        ins_conv0 = self.ins_firstconv(ins_feat)
        ins_conv0 = torch.cat([ins_conv0, skip2], dim=1)

        sem_conv1 = self.upsample(self.sem_conv1(sem_conv0))  # 1/8
        ins_conv1 = self.ins_conv1(ins_conv0)

        sem_conv2 = self.upsample(self.sem_conv2(sem_conv1))  # 1/4
        ins_conv2 = self.ins_conv2(ins_conv1)

        sem_conv2 = torch.cat([sem_conv2, skip1], dim=1)
        sem_conv3 = self.upsample(self.sem_conv3(sem_conv2))  # 1/2
        out_sem = self.upsample(self.sem_lastconv(sem_conv3))  # 1

        ins_conv2 = torch.cat([ins_conv2, skip1], dim=1)

        out_emb = self.ins_emb_conv1(ins_conv2)  # 1/2
        out_emb = self.ins_emb_conv2(out_emb)  # 1
        out_emb = self.ins_emb_lastconv(out_emb)  # 1

        out_cen = self.ins_cen_conv1(ins_conv2)  # 1/2
        out_cen = self.ins_cen_lastconv(out_cen)  # 1/2
        out_cen = self.upsample(out_cen)  # 1
        out_cen = torch.sigmoid(out_cen)

        return out_sem, out_emb, out_cen


class PanopticDecoderThreeBranch(nn.Module):
    def __init__(
            self,
            input_channels,
            n_layers=2,
            expand=2,
            num_classes=21,
            emb_dim=32,
            upsample_mode="conv",
    ):
        super(PanopticDecoderThreeBranch, self).__init__()

        assert upsample_mode in ["conv", "bilinear"]
        self.upsample_mode = upsample_mode

        self.upsample = torch.nn.Upsample(scale_factor=2,
                                          mode='bilinear',
                                          align_corners=False)

        """ Semantic Decoder """
        # first conv layer
        self.sem_firstconv = nn.Sequential(
            ConvNormActivation(input_channels, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU),
            ConvNormActivation(80, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=None)
        )
        # concat with skip2 80  stride = 1/16
        self.sem_conv1 = create_decoder_block_conv_ds(
            160, 80, expand=expand, n_layers=n_layers)
        # Followed by an up-sampling x2
        # stride = 1/8
        self.sem_conv2 = create_decoder_block_conv_ds(
            80, 40, expand=expand, n_layers=n_layers)
        # Followed by an up-sampling x2
        # concat with skip1 24  stride = 1/4
        self.sem_conv3 = create_decoder_block_conv_ds(
            64, 64, expand=expand, n_layers=n_layers)
        # Followed by up-sampling x2  sride = 1/2
        self.sem_lastconv = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(),
                                          nn.Conv2d(64, num_classes, 1))
        # Followed by up-sampling x2 to original resolution

        """ Instance Embedding Decoder"""
        self.emb_firstconv = nn.Sequential(
            ConvNormActivation(input_channels, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU),
            ConvNormActivation(80, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=None)
        )
        # concat with skip2 80  stride = 1/16
        # self.emb_conv1 = create_decoder_block_conv_ds(160, 80, expand=expand, n_layers=n_layers)
        self.emb_conv1 = ObjDecoderModule(160, 80, nr_decoder_blocks=n_layers)
        # stride = 1/8
        # self.emb_conv2 = create_decoder_block_conv_ds(80, 40, expand=expand, n_layers=n_layers)
        self.emb_conv2 = ObjDecoderModule(80, 40, nr_decoder_blocks=n_layers)
        # concat with skip1 24  stride = 1/4
        # self.emb_conv3 = create_decoder_block_conv_ds(64, emb_dim, expand=expand, n_layers=n_layers)
        self.emb_conv3 = ObjDecoderModule(
            64, 64, nr_decoder_blocks=n_layers, upsample=False)
        # embedding head stride = 1/4
        self.emb_lastconv = nn.Conv2d(64, emb_dim, 3, padding=1)
        # Followed by 2 upsample x2
        self.emb_up1 = Upsample(channels=emb_dim)
        self.emb_up2 = Upsample(channels=emb_dim)

        """ Instance Center Decoder"""
        # shared part
        self.cen_firstconv = nn.Sequential(
            ConvNormActivation(input_channels, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU),
            ConvNormActivation(80, 80, kernel_size=3,
                               norm_layer=nn.BatchNorm2d, activation_layer=None)
        )
        # concat with skip2 80  stride = 1/16
        # self.cen_conv1 = create_decoder_block_conv_ds(160, 80, expand=expand, n_layers=n_layers)
        self.cen_conv1 = ObjDecoderModule(160, 128, nr_decoder_blocks=n_layers)
        # stride = 1/8
        # self.cen_conv2 = create_decoder_block_conv_ds(80, 40, expand=expand, n_layers=n_layers)
        self.cen_conv2 = ObjDecoderModule(128, 104, nr_decoder_blocks=n_layers)
        # concat with skip1 24  stride = 1/4
        self.cen_conv3 = ObjDecoderModule(
            128, 128, nr_decoder_blocks=n_layers, upsample=False)
        # center head stride = 1/4
        self.cen_lastconv = nn.Conv2d(128, 1, 3, padding=1)
        # Followed by 2 upsample x2
        self.cen_up1 = Upsample(channels=1)
        self.cen_up2 = Upsample(channels=1)

    def forward(self, sem_feat, ins_feat, skip1, skip2):
        """
        :param sem_feat: feature output from semantic-ASPP
        :param ins_feat: feature output from inatance-ASPP
        :param skip1: feat skip1
        :param skip2: feat skip2
        :return:
        """
        sem_conv0 = self.sem_firstconv(sem_feat)  # 1/16
        sem_conv0 = torch.cat([sem_conv0, skip2], dim=1)
        sem_conv1 = self.upsample(self.sem_conv1(sem_conv0))  # 1/8
        sem_conv2 = self.upsample(self.sem_conv2(sem_conv1))  # 1/4
        sem_conv2 = torch.cat([sem_conv2, skip1], dim=1)
        sem_conv3 = self.upsample(self.sem_conv3(sem_conv2))  # 1/2
        out_sem = self.upsample(self.sem_lastconv(sem_conv3))  # 1

        emb_conv0 = self.emb_firstconv(ins_feat)  # 1/16
        emb_conv0 = torch.cat([emb_conv0, skip2], dim=1)
        emb_conv1 = self.emb_conv1(emb_conv0)  # 1/8
        emb_conv2 = self.emb_conv2(emb_conv1)  # 1/4
        emb_conv2 = torch.cat([emb_conv2, skip1], dim=1)
        emb_conv3 = self.emb_conv3(emb_conv2)
        out_emb = self.emb_lastconv(emb_conv3)
        out_emb = self.emb_up1(out_emb)  # 1/2
        out_emb = self.emb_up2(out_emb)  # 1

        cen_conv0 = self.cen_firstconv(ins_feat)  # 1/16
        cen_conv0 = torch.cat([cen_conv0, skip2], dim=1)
        cen_conv1 = self.cen_conv1(cen_conv0)  # 1/8
        cen_conv2 = self.cen_conv2(cen_conv1)  # 1/4
        cen_conv2 = torch.cat([cen_conv2, skip1], dim=1)
        cen_conv3 = self.cen_conv3(cen_conv2)
        out_cen = self.cen_lastconv(cen_conv3)
        out_cen = self.cen_up1(out_cen)  # 1/2
        out_cen = self.cen_up2(out_cen)  # 1
        out_cen = torch.sigmoid(out_cen)

        return out_sem, out_emb, out_cen


class Upsample(nn.Module):
    def __init__(self, mode="learned-3x3-zeropad", channels=None):
        super(Upsample, self).__init__()
        self.interp = F.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2] * 2), int(x.shape[3] * 2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


def create_decoder_block_conv_ds(input_channels, output_channels, expand=2, n_layers=2, bias=None):
    modules = [DecoderModuleConvDS(input_channels, input_channels * expand, output_channels,
                                   use_res_connect=input_channels == output_channels, bias=bias)]
    for _ in range(1, n_layers):
        modules.append(DecoderModuleConvDS(output_channels, output_channels * expand,
                                           output_channels, use_res_connect=True, bias=bias))
    return nn.Sequential(*modules)


class DecoderModuleConv(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernal_size=3,
            padding=1,
            use_res_connect=True,
            activation=nn.ReLU(),
    ):
        super().__init__()

        if use_res_connect:
            assert input_channels == output_channels, "input and output dimension should match!!!"
        self.activation = activation
        self.block = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=kernal_size, padding=padding),
                                   nn.BatchNorm2d(output_channels),
                                   activation,
                                   nn.Conv2d(input_channels, output_channels,
                                             kernel_size=kernal_size, padding=padding),
                                   nn.BatchNorm2d(output_channels))
        self.use_res_connect = use_res_connect

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return self.activation(result)


class ConvBNAct(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)


class ConvBN(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(ConvBN, self).__init__()
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2,
                                          bias=False))
        self.add_module('bn', nn.BatchNorm2d(channels_out))


class ObjDecoderModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_out,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=2,
                 upsample=True,
                 upsampling_mode='bilinear'):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.upsample = upsample

        self.conv3x3 = ConvBNAct(channels_in, channels_out, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_out,
                                          channels_out,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)
        # self.conv_out = ConvBN(channels_out, channels_out, 3)
        if self.upsample:
            if upsampling_mode == "bilinear":
                self.upsample = torch.nn.Upsample(scale_factor=2,
                                                  mode='bilinear',
                                                  align_corners=False)
            elif upsampling_mode == "nearest":
                self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            else:
                raise NotImplementedError

    def forward(self, decoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)
        if self.upsample:
            out = self.upsample(out)
        # out = self.conv_out(out)
        return out


class NonBottleneck1D(nn.Module):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super().__init__()

        dropprob = 0
        self.conv3x1_1 = nn.Conv2d(inplanes, planes, (3, 1),
                                   stride=(stride, 1), padding=(1, 0),
                                   bias=True)
        self.conv1x3_1 = nn.Conv2d(planes, planes, (1, 3),
                                   stride=(1, stride), padding=(0, 1),
                                   bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-03)
        self.act = activation
        self.conv3x1_2 = nn.Conv2d(planes, planes, (3, 1),
                                   padding=(1 * dilation, 0), bias=True,
                                   dilation=(dilation, 1))
        self.conv1x3_2 = nn.Conv2d(planes, planes, (1, 3),
                                   padding=(0, 1 * dilation), bias=True,
                                   dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)

        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output
        else:
            return self.act(output + identity)


# Decoder module based on depthwise separable conv i.e. expand -> depth-wise conv -> project
class DecoderModuleConvDS(nn.Module):
    def __init__(
            self,
            input_channels,
            expand_channels,
            output_channels,
            kernal_size=3,
            use_res_connect=False,
            activation=nn.ReLU,
            bias=None,
    ):
        super().__init__()

        if use_res_connect:
            assert input_channels == output_channels, "input and output dimension should match!!!"
        layers = list()
        # expand
        layers.append(
            ConvNormActivation(
                input_channels,
                expand_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=activation,
                bias=bias
            )
        )
        # depth-wise conv
        layers.append(
            ConvNormActivation(
                expand_channels,
                expand_channels,
                kernel_size=kernal_size,
                groups=expand_channels,
                norm_layer=nn.BatchNorm2d,
                activation_layer=activation,
                bias=bias
            )
        )
        # projection
        layers.append(
            ConvNormActivation(
                expand_channels,
                output_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None,
                bias=bias
            )
        )
        self.activation = activation()
        self.block = nn.Sequential(*layers)
        self.use_res_connect = use_res_connect

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class DecoderModuleConvTransposeDS(nn.Module):
    def __init__(
            self,
            input_channels,
            expand_channels,
            output_channels,
            kernal_size=3,
            stride=2,  # up-sampling x2
            use_res_connect=False,
            activation=nn.ReLU,
    ):
        super().__init__()

        if use_res_connect:
            assert input_channels == output_channels, "input and output dimension should match!!!"
        layers = list()
        # expand
        layers.append(
            ConvTransposeNormActivation(
                input_channels,
                expand_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=activation,
            )
        )
        # depth-wise conv
        layers.append(
            ConvTransposeNormActivation(
                expand_channels,
                expand_channels,
                kernel_size=kernal_size,
                stride=stride,
                groups=expand_channels,
                norm_layer=nn.BatchNorm2d,
                activation_layer=activation,
            )
        )
        # projection
        layers.append(
            ConvNormActivation(
                expand_channels,
                output_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=activation,  # activation,
            )
        )
        self.activation = activation()
        self.block = nn.Sequential(*layers)
        self.use_res_connect = use_res_connect

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return self.activation(result)


class ConvTransposeNormActivation(torch.nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            output_padding=None,
            groups=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            dilation=1,
            inplace=True,
            bias=None,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if output_padding is None:
            output_padding = stride - 1
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels
