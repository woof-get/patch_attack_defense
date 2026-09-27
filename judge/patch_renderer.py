"""裁判端贴片渲染 (apply_patch) - 对齐赛题 §12/§19。

职责:
- 采样渲染参数(位置/缩放/旋转/颜色抖动), 位置限制在目标框中央 60% 区域。
- apply_patch: 把攻击贴片按参数渲染到图像上。
- 自然遮挡基线: 用相同参数渲染 gray/black/white/noise/texture 贴片(§19 公平扣减遮挡)。
- apply_patch_torch: 可微版本(供梯度攻击反传)。

本模块不提供给防御方(赛题 §13/§65)。
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .utils import to_float01, to_uint8

# ----------------- 参数采样 -----------------

def sample_params(bbox, cfg, rng) -> Dict:
    """采样贴片渲染参数。bbox: [x1,y1,x2,y2] (图像坐标)。"""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    bbox_area = bw * bh
    a_cfg = cfg.attack
    max_ratio = float(getattr(a_cfg, "max_patch_area_ratio", 0.20))
    # 显示边长: 使贴片面积 = max_ratio * bbox_area, 再加缩放抖动
    base_side = math.sqrt(bbox_area * max_ratio)
    s_lo, s_hi = a_cfg.scale_range
    scale = float(rng.uniform(s_lo, s_hi))
    side = max(8, int(round(base_side * scale)))
    side = min(side, int(min(bw, bh)))  # 不超过目标框短边
    rot_lo, rot_hi = a_cfg.rotation_range
    angle = math.radians(float(rng.uniform(rot_lo, rot_hi)))
    inner = float(getattr(a_cfg, "position_inner_ratio", 0.60))
    # 中心限制在目标框中央 inner 区域
    cx_min = x1 + bw * (1 - inner) / 2
    cx_max = x2 - bw * (1 - inner) / 2
    cy_min = y1 + bh * (1 - inner) / 2
    cy_max = y2 - bh * (1 - inner) / 2
    cx = float(rng.uniform(cx_min, cx_max))
    cy = float(rng.uniform(cy_min, cy_max))
    cj = float(getattr(a_cfg, "color_jitter", 0.10))
    color_factor = float(rng.uniform(1 - cj, 1 + cj))
    return {"side": side, "angle": angle, "cx": cx, "cy": cy,
            "color_factor": color_factor, "bbox": [x1, y1, x2, y2]}


# ----------------- 自然遮挡贴片池 (§19) -----------------

def occlusion_patch(kind: str, size: int, rng) -> np.ndarray:
    """生成自然遮挡贴片 (3,size,size) float [0,1]。"""
    if kind == "gray":
        p = np.full((size, size, 3), 128, dtype=np.uint8)
    elif kind == "black":
        p = np.zeros((size, size, 3), dtype=np.uint8)
    elif kind == "white":
        p = np.full((size, size, 3), 255, dtype=np.uint8)
    elif kind == "noise":
        p = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    elif kind == "texture":
        # 棋盘纹理
        p = np.zeros((size, size, 3), dtype=np.uint8)
        c = rng.integers(0, 256, size=3)
        cell = max(2, size // 8)
        for i in range(0, size, cell):
            for j in range(0, size, cell):
                if ((i // cell) + (j // cell)) % 2 == 0:
                    p[i:i + cell, j:j + cell] = c
        p = p + rng.integers(-10, 10, p.shape)
        p = np.clip(p, 0, 255).astype(np.uint8)
    else:
        p = np.full((size, size, 3), 128, dtype=np.uint8)
    return to_float01(p)  # HWC float [0,1]


OCC_POOL = ["gray", "black", "white", "noise", "texture"]


# ----------------- 可微渲染 (torch, 供攻击 + 评测共用) -----------------

def render_patch_torch(image, patch, params, img_size):
    """可微贴片渲染。

    image: tensor (1,3,H,W) 或 (3,H,W) float [0,1]
    patch: tensor (3,ps,ps) float [0,1] (requires_grad)
    params: dict(side, angle, cx, cy, color_factor)
    返回: tensor (1,3,H,W) float [0,1], 与 patch 有计算图连接。
    """
    import torch
    import torch.nn.functional as F
    if image.ndim == 3:
        image = image.unsqueeze(0)
    B = image.shape[0]
    device = image.device
    side = int(params["side"])
    angle = float(params["angle"])
    cx, cy = float(params["cx"]), float(params["cy"])
    cf = float(params["color_factor"])

    # 1) 缩放贴片到 side x side, 应用颜色抖动
    p = patch.unsqueeze(0)  # (1,3,ps,ps)
    if side != patch.shape[-1]:
        p = F.interpolate(p, size=(side, side), mode="bilinear", align_corners=False)
    p = p * cf
    p = p.clamp(0, 1)

    # 2) 构建采样网格: 对输出图像每个像素, 求其在贴片局部归一化坐标 (u,v) in [-1,1]
    yy, xx = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32, device=device),
        torch.arange(img_size, dtype=torch.float32, device=device),
        indexing="ij")
    dx = xx - cx
    dy = yy - cy
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # 撤销旋转 (R(-angle)) 得到贴片局部坐标
    px = cos_a * dx + sin_a * dy
    py = -sin_a * dx + cos_a * dy
    half = side / 2.0
    u = px / half
    v = py / half
    grid = torch.stack([u, v], dim=-1).unsqueeze(0)  # (1,H,W,2)
    # 3) 采样贴片到图像画布
    sampled = F.grid_sample(p, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # (1,3,H,W)
    # 4) 软掩膜 (内部为 1, 边界平滑)
    mask_u = (u.abs() <= 1.0).float()
    mask_v = (v.abs() <= 1.0).float()
    # 边界软化: 用 1 - relu(|u|-1) 之类, 这里用硬掩膜 + grid_sample 的双线性已提供边缘过渡
    mask = (mask_u * mask_v).unsqueeze(0)  # (1,1,H,W)
    out = image * (1 - mask) + sampled * mask
    return out.clamp(0, 1)


# ----------------- 评测用便捷渲染 (uint8 HWC 输入输出) -----------------

def _to_patch_tensor(patch):
    import torch
    if isinstance(patch, torch.Tensor):
        t = patch.float()
        if t.ndim == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        if t.max() > 1.5:
            t = t / 255.0
        return t.clamp(0, 1)
    arr = to_float01(np.asarray(patch))  # HWC
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(np.ascontiguousarray(arr)).float().clamp(0, 1)


def render_patch(image_hwc, patch, params, img_size):
    """渲染贴片到 HWC uint8 图像 -> HWC uint8。"""
    import torch
    img_t = _to_patch_tensor(image_hwc).unsqueeze(0)
    p_t = _to_patch_tensor(patch)
    with torch.no_grad():
        out = render_patch_torch(img_t, p_t, params, img_size)
    return to_uint8(out[0].permute(1, 2, 0).cpu().numpy())


def apply_patch(image_hwc, bbox, patch, rng, cfg):
    """裁判端: 采样参数 + 渲染攻击贴片。返回 (patched_image_hwc, params)。"""
    img_size = int(getattr(cfg.dataset, "image_size", image_hwc.shape[0]))
    params = sample_params(bbox, cfg, rng)
    out = render_patch(image_hwc, patch, params, img_size)
    return out, params


def apply_occlusion(image_hwc, bbox, params, rng, cfg):
    """用相同参数渲染自然遮挡贴片 (§19)。随机选一种遮挡类型。"""
    img_size = int(getattr(cfg.dataset, "image_size", image_hwc.shape[0]))
    kind = rng.choice(OCC_POOL)
    occ = occlusion_patch(kind, int(params["side"]), rng)
    out = render_patch(image_hwc, occ, params, img_size)
    return out, kind


def apply_patch_batch(images_hwc, bboxes, patch, rng, cfg):
    """批量渲染 (每图独立采样参数)。images_hwc: list[HWC uint8]。返回 list[(patched, params)]。"""
    results = []
    for img, bbox in zip(images_hwc, bboxes):
        results.append(apply_patch(img, bbox, patch, rng, cfg))
    return results
