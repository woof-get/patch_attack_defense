"""防御运行器: 加载 DefenseModel 并批量推理出 patch_probability。

支持两种提交形式 (赛题 §29):
- 类: class DefenseModel: __init__(model_path); predict(image) -> float
- 函数: defend(env, images) -> ndarray[N]
"""
from __future__ import annotations

from typing import List

import numpy as np

from .env import DefenseEnv


class DefenseRunner:
    """包装一个防御方法, 提供统一 predict_many。"""

    def __init__(self, defense, model_path=None, env: DefenseEnv = None):
        self.kind = None  # 'class' | 'func' | 'instance'
        self.instance = None
        self.func = None
        if hasattr(defense, "predict") and not isinstance(defense, type):
            # 已训练的 DefenseModel 实例
            self.instance = defense
            self.kind = "instance"
        elif isinstance(defense, type):
            # 类: 实例化
            try:
                self.instance = defense(model_path) if model_path else defense()
            except TypeError:
                self.instance = defense(model_path)
            self.kind = "class"
            assert hasattr(self.instance, "predict"), "DefenseModel 必须实现 predict(image)->float"
        elif callable(defense):
            self.func = defense
            self.kind = "func"
        else:
            raise ValueError(f"不支持的防御形式: {type(defense)}")
        self.env = env

    def predict_many(self, images: List) -> np.ndarray:
        """images: list[HWC uint8] -> ndarray[N] float [0,1]。"""
        if self.kind in ("class", "instance"):
            out = []
            for img in images:
                p = self.instance.predict(img)
                out.append(float(np.clip(p, 0.0, 1.0)))
            return np.asarray(out, dtype=np.float32)
        else:
            probs = self.func(self.env, images)
            probs = np.asarray(probs, dtype=np.float32).reshape(-1)
            return np.clip(probs, 0.0, 1.0)

    def predict_one(self, image) -> float:
        if self.kind in ("class", "instance"):
            return float(np.clip(self.instance.predict(image), 0.0, 1.0))
        return float(np.clip(self.func(self.env, [image])[0], 0.0, 1.0))
