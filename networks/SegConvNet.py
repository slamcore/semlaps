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

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityNet(nn.Module):
    def __init__(self, hidden_unit=[16, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale = bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)

        return density_scale


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        # TODO: Think about how to encode co-variance and normal
        """
        :param in_channel: C=3
        :param out_channel: C_mid
        :param hidden_unit:
        """
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv1d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv1d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv1d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))

    def forward(self, localized_xyz):
        # xyz : NxCxK
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))
            # weights = F.relu(conv(weights))

        return weights


class SegConvLayer(nn.Module):
    def __init__(self,
                 input_channel,
                 weight_in="xyz",
                 mid_dim=16,
                 hidden_unit=[8, 8],
                 hidden_dims=[64, 64]):
        """
        :param input_channel: input feature dim
        :param weight_in: input type to weight_net supported: [xyz+normal, xyz+cov_dir, xyz+vi, xyz+vi+cov_dir]
        :param mid_dim: output_dim for weight_net
        :param hidden_unit: hidden dims for weight_net
        :param hidden_dims: hidden dims for feature_net
        """
        super().__init__()
        # Feature MLPs
        feature_mlps = []
        last_channel = input_channel
        for out_channel in hidden_dims:
            feature_mlps += [
                nn.Conv1d(last_channel, out_channel, 1),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            ]
            last_channel = out_channel
        self.feature_mlp = nn.Sequential(*feature_mlps)

        if weight_in == "xyz":
            self.weight_in_dim = 3
        elif weight_in == "xyz+normal":
            self.weight_in_dim = 6
        elif weight_in == "xyz+cov_dir":
            self.weight_in_dim = 5  # 3 + 2
        elif weight_in == "xyz+vi":
            self.weight_in_dim = 12  # 3 + 9
        elif weight_in == "xyz+vi+cov_dir":
            self.weight_in_dim = 14  # 3 + 9 + 2
        else:
            raise NotImplementedError

        self.weight_in = weight_in
        self.weight_mlp = WeightNet(self.weight_in_dim, mid_dim, hidden_unit=hidden_unit)
        self.linear_layer = nn.Linear(mid_dim * last_channel, last_channel)  # to get final feature
        self.bn_linear = nn.BatchNorm1d(last_channel)

    def forward(self, xyz, cov, feats, knn_indices):
        """
        :param xyz: raw xyz position [B, N, 3], i.e. not subtracted by the center point
        :param cov: covariance matrices [B, N, 3, 3]
        :param feats: feature vectors associated with each segment_v1 [B, N, C]
        :param knn_indices: knn_matrix [N, K]
        Note B should always be 1
        :return:
        """
        # B is always 1
        # xyz: [B, N, 3]
        # cov: [B, N, 3, 3]
        # feats: [B, N, C]
        xyz = xyz.squeeze()  # [N, 3]
        cov = cov.squeeze()  # [N, 3, 3]
        feats = feats.squeeze()  # [N, C]
        N, C = feats.shape
        _, K = knn_indices.shape

        # get knn locations and features
        knn_indices_flatten = knn_indices.reshape(N * K)
        xyz = xyz[knn_indices_flatten, :].view(N, K, 3)
        localized_xyz = xyz - xyz[:, 0, :].unsqueeze(1)  # [N, K, 3]
        cov = cov[knn_indices_flatten, :, :].view(N, K, 3, 3)  # [N, K, 3, 3]
        feats = feats[knn_indices_flatten, :].view(N, K, C)  # [N, K, C]
        norm = feats[:, :, 3:6]  # [N, K, 3]

        if self.weight_in == "xyz":
            weightnet_input = localized_xyz.permute(0, 2, 1)  # [N, 3, K]
        elif self.weight_in == "xyz+normal":
            weightnet_input = torch.cat([localized_xyz.permute(0, 2, 1), norm.permute(0, 2, 1)], dim=1)
        elif self.weight_in == "xyz+cov_dir":
            sigma1 = torch.bmm(localized_xyz.reshape(-1, 1, 3), cov.reshape(-1, 3, 3))  # [NK, 1, 3]
            sigma1 = torch.bmm(sigma1, localized_xyz.reshape(-1, 3, 1)).squeeze(-1).view(N, K, 1).permute(0, 2, 1)  # [N, 1, K]
            cov0 = cov[:, 0, :, :].unsqueeze(1).repeat(1, K, 1, 1)  # [N, K, 3, 3]
            sigma2 = torch.bmm(localized_xyz.reshape(-1, 1, 3), cov0.reshape(-1, 3, 3))  # [NK, 1, 3]
            sigma2 = torch.bmm(sigma2, localized_xyz.reshape(-1, 3, 1)).squeeze(-1).view(N, K, 1).permute(0, 2, 1)
            weightnet_input = torch.cat([localized_xyz.permute(0, 2, 1), sigma1, sigma2], dim=1)
        else:  # In all the rest cases we need to compute VI features
            # Taken from VI-Pointconv, following their dimension
            n_alpha = norm  # [N, K, 3]
            n_miu = n_alpha[:, 0, :]  # [N, 3]
            r_miu = localized_xyz  # [N, K, 3]
            r_hat = F.normalize(r_miu, dim=2)  # [N, K, 3]
            # [N, K, 3] = ([N, 1, 3] @ [N, 3, K]).permute(0, 2, 1) * [N, K, 3]
            v_miu = n_miu.unsqueeze(dim=1) - torch.matmul(n_miu.unsqueeze(dim=1), r_hat.permute(0, 2, 1)).permute(0, 2, 1) * r_hat
            v_miu = F.normalize(v_miu, dim=2)
            w_miu = torch.cross(r_hat, v_miu, dim=2)
            w_miu = F.normalize(w_miu, dim=2)
            # compute VI features and reshape back to our dimesnion, i.e. [N, K, C] -> [N, C, K]
            theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=2)).permute(0, 2, 1)  # [N, 1, K]
            theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=2)).permute(0, 2, 1)  # [N, 1, K]
            theta3 = torch.sum(r_hat * n_alpha, dim=2, keepdim=True).permute(0, 2, 1)
            theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=2)).permute(0, 2, 1)
            theta5 = torch.sum(n_alpha * r_hat, dim=2, keepdim=True).permute(0, 2, 1)
            theta6 = torch.sum(n_alpha * v_miu, dim=2, keepdim=True).permute(0, 2, 1)
            theta7 = torch.sum(n_alpha * w_miu, dim=2, keepdim=True).permute(0, 2, 1)
            theta8 = torch.sum(r_miu * torch.cross(n_alpha, n_miu.unsqueeze(dim=1).repeat(1, K, 1), dim=2), dim=2, keepdim=True).permute(0, 2, 1)
            theta9 = torch.norm(r_miu, dim=2, keepdim=True).permute(0, 2, 1)

            if self.weight_in == "xyz+vi":
                weightnet_input = torch.cat([localized_xyz.permute(0, 2, 1), theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9], dim=1)
            elif self.weight_in == "xyz+vi+cov_dir":
                sigma1 = torch.bmm(localized_xyz.reshape(-1, 1, 3), cov.reshape(-1, 3, 3))  # [NK, 1, 3]
                sigma1 = torch.bmm(sigma1, localized_xyz.reshape(-1, 3, 1)).squeeze(-1).view(N, K, 1).permute(0, 2, 1)  # [N, 1, K]
                cov0 = cov[:, 0, :, :].unsqueeze(1).repeat(1, K, 1, 1)  # [N, K, 3, 3]
                sigma2 = torch.bmm(localized_xyz.reshape(-1, 1, 3), cov0.reshape(-1, 3, 3))  # [NK, 1, 3]
                sigma2 = torch.bmm(sigma2, localized_xyz.reshape(-1, 3, 1)).squeeze(-1).view(N, K, 1).permute(0, 2, 1)
                weightnet_input = torch.cat([localized_xyz.permute(0, 2, 1),
                                             theta1, theta2, theta3, theta4, theta5,
                                             theta6, theta7, theta8, theta9, sigma1, sigma2], dim=1)
            else:
                raise NotImplementedError

        # [N, C_mid, K]
        weights = self.weight_mlp(weightnet_input)
        # [N, C_out, K]
        feats = feats.permute(0, 2, 1)
        feats = self.feature_mlp(feats)
        # [N, C_mid * C_out] = [N, C_out, K] @ [N, K, C_mid]
        new_feats = torch.matmul(feats, weights.permute(0, 2, 1)).view(N, -1)
        # [N, C_out]
        new_feats = F.relu(self.bn_linear(self.linear_layer(new_feats)))
        # new_feats = F.relu(new_feats.permute(0, 2, 1)).permute(0, 2, 1)

        return new_feats


class SegConvNet(nn.Module):
    def __init__(self, input_feat_dim=30, weight_in="xyz", dropout_p=0.0, classifier_hidden_dims=[128, 64], num_classes=21):
        super().__init__()
        print("Using {}".format(weight_in))
        self.sv1 = SegConvLayer(input_channel=input_feat_dim, weight_in=weight_in, mid_dim=16, hidden_unit=[8, 8], hidden_dims=[64, 64])
        self.sv2 = SegConvLayer(input_channel=64, weight_in=weight_in, mid_dim=16, hidden_unit=[8, 8], hidden_dims=[128, 128])
        self.sv3 = SegConvLayer(input_channel=128, weight_in=weight_in, mid_dim=16, hidden_unit=[8, 8], hidden_dims=[128, 128])  # was [256, 256]
        # classifier
        classifier_layers = []
        in_dim = 128
        for feat_dim in classifier_hidden_dims:
            layer = [nn.Linear(in_dim, feat_dim),
                     nn.BatchNorm1d(num_features=feat_dim),
                     nn.ReLU()]
            if dropout_p > 0.0:
                layer.append(nn.Dropout(dropout_p))
            classifier_layers += layer
            in_dim = feat_dim
        classifier_layers += [nn.Linear(classifier_hidden_dims[-1], num_classes)]
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, xyz, cov, feats, knn_indices):
        feat1 = self.sv1(xyz, cov, feats, knn_indices)
        feat2 = self.sv2(xyz, cov, feat1, knn_indices)
        feat3 = self.sv3(xyz, cov, feat2, knn_indices)  # [N, feat_dim]
        out = self.classifier(feat3)  # [N, num_classes]

        return out.unsqueeze(0).permute(0, 2, 1)
