"""攻击方法基类。

接口 (赛题 §15-§18):
    attack(env, task) -> {"patch": ndarray (3, patch_size, patch_size) float [0,1]}

task = {
    "images":              list[HWC uint8]  攻击优化训练图像
    "annotations":         list[list[{bbox, category_id}]]  每图目标框
    "target_class":        int   目标 COCO 类别 id
    "patch_size":          int
    "max_patch_area_ratio":float
    "detector":            env.detector   公开 Model-A (predict/loss)
}

攻击方只返回 patch, 不返回改图/位置/缩放/旋转 (裁判统一渲染)。
"""
from __future__ import annotations

import numpy as np


class BaseAttack:
    """攻击方法基类。子类实现 __call__。"""

    name = "base"
    meta: dict = {}

    def __call__(self, env, task) -> dict:
        raise NotImplementedError

    @staticmethod
    def init_patch(patch_size: int, rng=None, mode="noise") -> np.ndarray:
        """初始化贴片 (3,ps,ps) float [0,1]。"""
        if rng is None:
            rng = np.random.default_rng()
        if mode == "noise":
            p = rng.random((3, patch_size, patch_size), dtype=np.float32)
        elif mode == "gray":
            p = np.full((3, patch_size, patch_size), 0.5, dtype=np.float32)
        elif mode == "checker":
            p = np.zeros((3, patch_size, patch_size), dtype=np.float32)
            cell = max(2, patch_size // 8)
            for i in range(0, patch_size, cell):
                for j in range(0, patch_size, cell):
                    if ((i // cell) + (j // cell)) % 2 == 0:
                        p[:, i:i + cell, j:j + cell] = 1.0
        else:
            p = rng.random((3, patch_size, patch_size), dtype=np.float32)
        return p


def as_attack(obj):
    """归一化为 BaseAttack 类 (供 registry 存储, evaluator 实例化)。"""
    if isinstance(obj, type) and issubclass(obj, BaseAttack):
        return obj
    if callable(obj):

        class _FuncAttack(BaseAttack):
            name = getattr(obj, "name", getattr(obj, "__name__", "attack"))

            def __init__(self, **kwargs):
                super().__init__()

            def __call__(self, env, task):
                return obj(env, task)

        return _FuncAttack
    raise ValueError(f"无法识别的攻击: {obj}")
