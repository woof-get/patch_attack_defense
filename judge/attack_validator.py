"""攻击合法性检查 (§18)。

- 18.1 输出尺寸: patch.shape == (3, patch_size, patch_size)  (CHW; 兼容 HWC 自动转)
- 18.2 像素范围: np.isfinite(patch).all() 且裁剪到 [0,1]
- 18.3 仅使用 patch 字段, 忽略额外输出
- 18.4 Universal 约束由 evaluator 保证(每轮只调一次 attack)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def validate_patch(result: dict, patch_size: int) -> Tuple[bool, np.ndarray, list]:
    """校验攻击输出。返回 (ok, patch_chw_float01, messages)。"""
    msgs = []
    if not isinstance(result, dict) or "patch" not in result:
        return False, None, ["输出必须为 dict 且包含 'patch' 字段"]
    patch = result["patch"]
    if isinstance(patch, list):
        patch = np.asarray(patch)
    if not isinstance(patch, np.ndarray):
        try:
            import torch
            if isinstance(patch, torch.Tensor):
                patch = patch.detach().cpu().numpy()
            else:
                return False, None, [f"patch 类型不支持: {type(patch)}"]
        except Exception:
            return False, None, [f"patch 类型不支持: {type(patch)}"]

    if patch.ndim != 3:
        return False, None, [f"patch 维度应为 3, 实际 {patch.ndim}"]

    # 统一到 CHW
    if patch.shape[-1] == 3 and patch.shape[0] != 3:
        patch = np.transpose(patch, (2, 0, 1))
    if patch.shape[0] != 3:
        return False, None, [f"patch 通道数应为 3, 实际 shape {patch.shape}"]

    H, W = patch.shape[1], patch.shape[2]
    if H != patch_size or W != patch_size:
        return False, None, [f"patch 尺寸应为 (3,{patch_size},{patch_size}), 实际 (3,{H},{W})"]

    if not np.isfinite(patch).all():
        msgs.append("警告: patch 含非有限值, 已替换为 0")
        patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)

    patch = patch.astype(np.float32)
    if patch.max() > 1.5:  # 误用 uint8 [0,255]
        msgs.append("警告: patch 似乎为 [0,255], 已归一化到 [0,1]")
        patch = patch / 255.0
    patch = np.clip(patch, 0.0, 1.0)

    extra = [k for k in result.keys() if k != "patch"]
    if extra:
        msgs.append(f"忽略额外字段: {extra}")
    return True, patch, msgs
