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
from torch.nn import functional as F
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)


def transform_points(points, T):
    """
    :param points:  [N, 3]
    :param T: [4, 4]
    :return:
    """
    points = (T[:3, :3] @ points.t() + T[:3, 3].unsqueeze(-1)).t()
    return points


def project_pcd(points, K, H, W, depth=None, crop=0, w2c=None, eps=0.05):
    """
    :param points: [N, 3]
    :param K: [3, 3]
    :param w2c: [4, 4]
    :param crop: crop the border
    :return:
    """
    if w2c is not None:
        points = transform_points(points, w2c)  # [N, 3]

    uvz = (K @ points.t()).t()  # [N, 3]
    pz = uvz[:, 2] + 1e-7
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # check in frustum
    valid_mask = (crop <= px) & (px <= W - 1 - crop) & (crop <= py) & (py <= H - 1 - crop) & (pz > 0)
    # check not occluded
    if depth is not None:
        u = torch.clip(px, crop, W - 1 - crop).long()
        v = torch.clip(py, crop, H - 1 - crop).long()
        # Missing-depth pixels will always be False
        valid_mask = valid_mask & (pz < (depth[v, u] + eps))

    uv_norm = torch.stack([2 * px / (W - 1.) - 1., 2 * py / (H - 1.) - 1.], dim=-1)

    return uv_norm, valid_mask


def transform_points_batch(points, T):
    """
    apply transform T (pre-multiply) to point cloud (both are batched)
    :param points:  [B, N, 3]
    :param T: [B, 4, 4]
    :return:  [B, N, 3]
    """
    points = (torch.bmm(T[:, :3, :3], points.permute(
        0, 2, 1)) + T[:, :3, 3].unsqueeze(-1)).permute(0, 2, 1)
    return points


def project_pcd_batch(points, K, H, W, depth=None, crop=0, w2c=None, eps=0.05):
    """
    Project the same point cloud to B different poses
    :param points: [B, N, 3]
    :param K: [B, 3, 3]
    :param w2c: [B, 4, 4]
    :param depth: [B, H, W]
    :param crop: crop the border
    :return:
    """

    B, N, _ = points.shape

    if w2c is not None:
        points = transform_points_batch(points, w2c)

    uvz = (K @ points.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, 3]
    pz = uvz[:, :, 2] + 1e-7
    px = uvz[:, :, 0] / pz
    py = uvz[:, :, 1] / pz

    # check in frustum
    valid_mask = (crop <= px) & (px <= W - 1 - crop) & (crop <= py) & (py <= H - 1 - crop) & (pz > eps)  # [B, N]

    # check not occluded
    if depth is not None:
        u = torch.clip(px, crop, W - 1 - crop).long()  # [B, N]
        v = torch.clip(py, crop, H - 1 - crop).long()  # [B, N]
        # Missing-depth pixels will always be False

        # batched query
        bs = torch.arange(B).unsqueeze(-1).repeat(1, N)  # [B, N]
        occ_mask = pz < (depth[bs, v, u] + eps)
        valid_mask = valid_mask & occ_mask

    uv_norm = torch.stack(
        [2 * px / (W - 1.) - 1., 2 * py / (H - 1.) - 1.], dim=-1)  # [B, N, 2]

    return uv_norm, valid_mask


def backproj_featmap_batch(depth, K, c2w, convention="OpenCV"):
    """
    Args:
        depth: [B, H, W]
        K: [B, 3, 3]
        c2w: [B, 4, 4]
        convention: ["OpenCV", "OpenGL"]

    Returns:
        points under world coordinate
    """
    device = K.device
    # depth = depth.squeeze(1)  # make sure it is B by H by W
    B, H, W = depth.shape
    fx, fy, cx, cy = K[:, 0, 0].view(B, 1, 1), K[:, 1, 1].view(
        B, 1, 1), K[:, 0, 2].view(B, 1, 1), K[:, 1, 2].view(B, 1, 1)  # [B, 1, 1]
    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(
        H, device=device), indexing="xy")  # [1, H, W]
    i, j = i.unsqueeze(0), j.unsqueeze(0)
    if convention == "OpenCV":
        dirs = torch.stack([(i - cx) / fx, (j - cy) / fy,
                            torch.ones(B, H, W, device=device)], -1)  # [B, H, W, 3]
    elif convention == "OpenGL":
        dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -
        torch.ones(B, H, W, device=device)], -1)  # [B, H, W, 3]
    else:
        raise NotImplementedError

    valid_mask = depth > 0.  # [B, H, W]
    pts_cam = (dirs * depth[..., None]).view(B, H * W, 3)  # [B, H*W, 3]
    pts_world = (c2w[:, :3, :3] @ pts_cam.permute(0, 2, 1) +
                 c2w[:, :3, 3].unsqueeze(-1)).permute(0, 2, 1).view(B, H, W, 3)

    return pts_world, valid_mask


def backproj_batch_py3d(depth, K, convention="OpenCV"):
    """
    Args:
        depth: [B, H, W]
        K: [B, 3, 3]
        convention: ["OpenCV", "OpenGL"]

    Returns:
    padded back-projected point clouds (batch)
    """
    device = K.device
    # depth = depth.squeeze()  # make sure it is [B, H, W]
    B, H, W = depth.shape
    fx, fy, cx, cy = K[:, 0, 0].view(B, 1, 1), K[:, 1, 1].view(
        B, 1, 1), K[:, 0, 2].view(B, 1, 1), K[:, 1, 2].view(B, 1, 1)  # [B, 1, 1]
    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(
        H, device=device), indexing="xy")  # [1, H, W]
    i, j = i.unsqueeze(0), j.unsqueeze(0)
    if convention == "OpenCV":
        dirs = torch.stack([(i - cx) / fx, (j - cy) / fy,
                            torch.ones(B, H, W, device=device)], -1)  # [B, H, W, 3]
    elif convention == "OpenGL":
        dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -
        torch.ones(B, H, W, device=device)], -1)  # [B, H, W, 3]
    else:
        raise NotImplementedError

    pcd = dirs * depth[..., None]  # [B, H, W, 3]
    pcd = pcd.view(B, -1, 3)  # [B, H*W, 3]
    valid_mask = depth.view(B, -1) > 0.  # [B, H*W]
    padded_pcd = get_padded_features(pcd, valid_mask)

    return padded_pcd, valid_mask


def get_padded_features(feats, masks):
    """
    :param feats: [B, N, C]
    :param mask: [B, N], True -> nonzero, False -> zero
    :return: [B, N, C], zeros are padded at the end
    """
    B, _, _ = feats.shape
    padded_feats = torch.zeros_like(feats)
    # TODO: get rid of this for-loop
    for i in range(B):
        feat, mask = feats[i], masks[i]
        padded_feats[i, :torch.count_nonzero(mask), :] = feat[mask, :]
    return padded_feats


# @torch.no_grad()
def warp_feature_batch(src_feat_map, depth_ref, K_ref, c2w_ref, w2c_src, K_src):
    """
    warp feature from other-view (src) to ref-view (ref) via simple warping:
    i.e. for each pixel in ref-view query the feature vector in src-view via
    back-projection, transformation and re-projection
    :param src_feat_map: [B, C, H, W]
    :param depth_ref: [B, 1, H, W]
    :param K_ref: [B, 3, 3]
    :param c2w_ref: [B, 4, 4]
    :param w2c_src: [B, 4, 4]
    :param K_src: [B, 3, 3]
    :return: [B, C, H, W]
    """
    B, C, H, W = src_feat_map.shape
    # [B, H, W, 3], [B, H, W]
    pts, valid_depth_mask = backproj_featmap_batch(
        depth_ref.squeeze(1), K_ref, c2w_ref)
    # [B, H, W, C]
    warped_feat_valid = torch.zeros(B, H, W, C).to(src_feat_map)
    # [B, H*W, 2], [B, H*W]
    uv, valid_proj_mask = project_pcd_batch(
        pts.view(B, H * W, 3), K_src, H, W, depth=None, crop=0, w2c=w2c_src)
    # [B, C, H_in, W_in] AND [B, H_out, W_out, 2] -> [B, C, H_out, W_out]
    # [1, C, H, W] AND [1, 1, H*W, 2] -> [1, C, 1, H*W]
    feat = F.grid_sample(src_feat_map, uv.unsqueeze(1)).squeeze(
        2).permute(0, 2, 1).view(B, H, W, -1)  # [B, H*W, C]
    # [B, H, W, C]
    warped_image = torch.zeros(B, H, W, C).to(src_feat_map)
    valid_proj_mask = valid_proj_mask.view(B, H, W)
    warped_feat_valid[valid_proj_mask, :] = feat[valid_proj_mask, :]
    warped_image[valid_depth_mask, :] = warped_feat_valid[valid_depth_mask]

    # [B, C, H, W]
    return warped_image.permute(0, 3, 1, 2)


def warp_feature_batch_render(src_feat_map, depth_src, K_ref, w2c_ref, c2w_src, K_src, radius=0.05):
    """
    warp feature from other-view (src) to ref-view (ref) via differentiable rendering:
    i.e. back-project the feature point cloud from src-view and then render it under ref-view
    :param src_feat_map: [B, C, H, W]
    :param depth_src: [B, 1, H, W]
    :param K_ref: [B, 3, 3]
    :param w2c_ref: [B, 4, 4]
    :param c2w_src: [B, 4, 4]
    :param K_src: [B, 3, 3]
    :return:
    """
    device = src_feat_map.device
    B, C, H, W = src_feat_map.shape
    pcd_src, mask = backproj_batch_py3d(
        depth_src.squeeze(1), K_src)  # [B, HW, 3], [B, HW]
    T = torch.bmm(w2c_ref, c2w_src)  # src2ref [B, 4, 4]
    pcd_ref = transform_points_batch(pcd_src, T)
    feats = get_padded_features(src_feat_map.view(
        B, C, -1).permute(0, 2, 1), mask)  # [B, HW, C]

    # re-render using pytorch3d
    pcd_py3d = Pointclouds(points=pcd_ref, features=feats)
    R = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)  # [B, 3, 3]
    t = torch.zeros(1, 3).repeat(B, 1)  # [B, 3]
    fx, fy, cx, cy = K_ref[:, 0, 0], K_ref[:,
                                     1, 1], K_ref[:, 0, 2], K_ref[:, 1, 2]
    f = torch.stack([fx, fy], dim=-1)  # [B, 2]
    p = torch.stack([cx, cy], dim=-1)  # [B, 2]
    img_size = (H, W)
    cameras = PerspectiveCameras(device=device, focal_length=-f,
                                 principal_point=p, image_size=(img_size,), R=R, T=t, in_ndc=False)
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=radius,
        points_per_pixel=10,
        max_points_per_bin=20000,
    )
    rasterizer = PointsRasterizer(
        cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    warped_feat_map = renderer(pcd_py3d)  # [B, H, W, C]

    # [B, C, H, W]
    return warped_feat_map.permute(0, 3, 1, 2)
