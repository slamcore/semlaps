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

from collections import OrderedDict
from typing import Optional, Dict, List, Tuple, Union
from functools import partial
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from networks.MobileNet import mobilenet_v3_large, ConvNormActivation
from networks.conv_module import depthwise_separable_conv
from networks.LatentPriorNetwork import SSMA, ASPP, FusionFeatures, warp_feature_batch_render, warp_feature_batch


class MVCNet(nn.Module):
    """
    This is our self-implemented multiview MVCNet model that supports:
    Multi-view RGBD (default): rgb and depth fusion with SSMA + feature warping (w/ depth, camera poses and K)
    """
    def __init__(self, num_classes=21, pretrained=True, modality="rgbd", reproject=True, use_ssma=True,
                 decoder_in_dim=256, decoder_feat_dim=256, decoder_head_dim=128, window_size=3,
                 projection_dims=[128, 32, 64], render=False, fusion_mode="max"):
        super(MVCNet, self).__init__()
        assert modality in ["rgb", "rgbd"], "Unknown modality type!!!"
        self.modality = modality  # rgbd or rgb
        self.use_depth = self.modality == "rgbd"  # use_depth == True means that depth is required either for feature extraction OR feature reprojection
        self.use_ssma = use_ssma  # use_ssma == True means feature is extracted from RGB+depth
        self.reproject = reproject  # whether to perform feature extraction or not
        # window_size > 1 doesn't make sense for reproject == False

        # original deeplab model
        self.rgb_encoder = mobilenet_v3_large(pretrained=pretrained, input_channels=3, dilated=True)
        if self.use_ssma:
            self.depth_encoder = mobilenet_v3_large(pretrained=pretrained, input_channels=1, dilated=True)
            # fuse (RGB and D) output of stage 1, 2, 5, stage 1 and 2 will also be used in skip connection
            self.fusion_layer1 = SSMA(self.rgb_encoder.stage1[-1].out_channels, compression_rate=6)
            self.fusion_layer2 = SSMA(self.rgb_encoder.stage2[-1].out_channels, compression_rate=6)
            self.fusion_layer3 = SSMA(self.rgb_encoder.stage4[-1].out_channels, compression_rate=6)
        bneck_feat_dim = self.rgb_encoder.lastconv.out_channels
        self.ASPP = ASPP(bneck_feat_dim, [12, 24, 36])

        # TODO: we should have project_dims[0] == decoder_in_dim
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

        self.render = render  # use py3d rendering or simple warping, only works when multi_view==True
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
        self.upsample = torch.nn.Upsample(scale_factor=2,
                                          mode='bilinear',
                                          align_corners=False)
        self.decoder = MVCNetDecoder(decoder_in_dim,
                                     decoder_channels=decoder_feat_dim,
                                     skip_dims=projection_dims[1:])
        # output head
        self.semantic_head = nn.Sequential(
            depthwise_separable_conv(decoder_feat_dim, decoder_head_dim, 5, padding=2),
            nn.Conv2d(decoder_head_dim, num_classes, 1)
        )
        # Followed by 2x upsample x2

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
        depth0 = self.depth_encoder.forward_first_conv(depth_input) if depth_input is not None else None
        # stage 1: 24, 1/4
        rgb1 = self.rgb_encoder.forward_stage1(rgb0)
        depth1 = self.depth_encoder.forward_stage1(depth0) if depth0 is not None else None
        fuse1 = self.fusion_layer1(rgb1, depth1) if self.use_ssma else rgb1  # 24, 1/4, no ReLU
        # stage 2: 40, 1/8
        rgb2 = self.rgb_encoder.forward_stage2(fuse1)
        depth2 = self.depth_encoder.forward_stage2(depth1) if depth1 is not None else None
        fuse2 = self.fusion_layer2(rgb2, depth2) if self.use_ssma else rgb2  # 40, 1/8, no ReLU
        # stage 3: 80, 1/16
        rgb3 = self.rgb_encoder.forward_stage3(fuse2)
        depth3 = self.depth_encoder.forward_stage3(depth2) if depth2 is not None else None
        # stage 4: 160, 1/16
        rgb4 = self.rgb_encoder.forward_stage4(rgb3)
        depth4 = self.depth_encoder.forward_stage4(depth3) if depth3 is not None else None
        fuse3 = self.fusion_layer3(rgb4, depth4) if self.use_ssma else rgb4  # 160, 1/16
        # stage 5: 160, 1/16
        rgb5 = self.rgb_encoder.forward_stage5(fuse3)
        # depth5 = self.depth_encoder.forward_stage5(depth4)

        # early feature
        early_features = self.rgb_encoder.lastconv(rgb5)  # 960, H/16, W/16
        # late feature
        late_features = self.ASPP(early_features)  # 256, H/16, W/16 after ReLU

        # Feature maps used in decoder
        # bneck feature
        bneck_feat = self.project_feat(late_features)  # 128, H/16, W/16, after ReLU
        skip1 = self.project_skip1(fuse1)  # 32, H/4, W/4, after ReLU
        skip2 = self.project_skip2(fuse2)  # 64, H/8, W/8, after ReLU

        # decoder
        last_feat = self.decoder(bneck_feat, skip1, skip2)

        return last_feat

    def single_view_forward(self, rgb_input, depth_input=None) -> Dict[str, Tensor]:
        """
        single_view forward pass
        :param rgb_input: [B, 3, H, W]
        :param depth_input: [B, 1, H, W] or None
        :return:
        """

        feat = self.feature_net_forward(rgb_input, depth_input=depth_input)
        label = self.upsample(self.upsample(self.semantic_head(feat)))
        ret = {
            "out": label,
            "feat": feat
        }
        return ret

    def forward(self, rgb_input, depth_input, K_list, c2w_list, w2c_list, aux_loss=False, p_drop_depth=0.0) -> Dict[str, Tensor]:
        """
        multi_view forward pass
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
        ref_feat = self.feature_net_forward(rgb_input[:, 0, ...], depth_input=None if drop_depth_flag else depth_input[:, 0, ...])
        feat = [ref_feat]

        # depth_ref = depth_input[:, 0, ...]
        # c2w_ref = c2w_list[:, 0, ...]
        if aux_loss:
            feat_aux = [ref_feat]

        for n in range(1, self.window_size):
            # get multi-level features of the other views
            src_feat = self.feature_net_forward(rgb_input[:, n, ...], depth_input=None if drop_depth_flag else depth_input[:, n, ...])
            # for auxiliary output
            if aux_loss:
                feat_aux.append(src_feat)

            # get warped features
            K1 = K_list.clone()
            K1[:, 0:2, :] /= 4.
            if self.render:
                depth_scaled = F.interpolate(depth_input[:, n, ...], scale_factor=1/4., mode="nearest")
                warped_feat = warp_feature_batch_render(src_feat, depth_scaled, K1, w2c_list[:, 0, ...],
                                                        c2w_list[:, n, ...], K1, radius=self.render_radius[H // 4])
            else:
                depth_scaled = F.interpolate(depth_input[:, 0, ...], scale_factor=1/4., mode="nearest")
                warped_feat = warp_feature_batch(src_feat, depth_scaled, K1,
                                                 c2w_list[:, 0, ...],
                                                 w2c_list[:, n, ...], K1)
            feat.append(warped_feat)

        # concat and project
        if self.fusion_mode == "concat":
            fused_feat = self.fuse_feat(torch.cat(feat, dim=1))  # N * [B, C, H/4, W/4] -> [B, C, H/4, W/4]
        else:
            fused_feat = self.fuse_feat(feat)  # N * [B, C, H/4, W/4] -> [B, C, H/4, W/4]

        result = OrderedDict()
        # output: num_classes, 1
        out_sem = self.upsample(self.upsample(self.semantic_head(fused_feat)))

        result["out"] = out_sem
        result["feat"] = fused_feat

        if aux_loss:
            # must first stack along dim1 and then squeeze dim0 and dim1
            feat_aux = torch.stack(feat_aux, dim=1).view(B * N, -1, H // 4, W // 4)  # [BN, C, H/4, W/4]
            aux_out_sem = self.upsample(self.upsample(self.semantic_head(feat_aux)))
            result["aux"] = aux_out_sem

        return result


class MVCNetDecoder(nn.Module):
    """
    Decoder follows Panoptic-DeepLab architecture
    """
    def __init__(
            self,
            input_channels,  # output channel from ASPP
            decoder_channels=256,
            skip_dims=[32, 64],
    ):
        super(MVCNetDecoder, self).__init__()

        self.upsample = torch.nn.Upsample(scale_factor=2,
                                          mode='bilinear',
                                          align_corners=False)

        # upsample x2 -> 256, 1/8

        # concat with skip2 40  stride = 1/8
        self.conv1 = depthwise_separable_conv(input_channels + skip_dims[1], decoder_channels, 5, padding=2)

        # concat with skip1 24  stride = 1/4
        self.conv2 = depthwise_separable_conv(decoder_channels + skip_dims[0], decoder_channels, 5, padding=2)

    def forward(self, feat, skip1, skip2):
        # 256, 1/8
        conv0 = self.upsample(feat)
        conv0 = torch.cat([conv0, skip2], dim=1)
        # 256, 1/4
        conv1 = self.upsample(self.conv1(conv0))
        conv1 = torch.cat([conv1, skip1], dim=1)
        conv2 = self.conv2(conv1)
        return conv2
