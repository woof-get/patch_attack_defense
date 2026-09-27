"""Env: 传递给攻击/防御方法的环境对象。

为强制"攻防解耦":
- AttackEnv: 暴露公开检测器 Model-A (前向+梯度) 与可微贴片渲染器 (供 EOT 优化)。
- DefenseEnv: 仅暴露图像尺寸/设备等无害信息, 不暴露检测器/渲染器/贴片。
防御方拿到的是 DefenseEnv, 从 API 层面无法访问检测器或官方渲染逻辑。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import Config


class AttackEnv:
    """攻击方环境: 公开检测器 + 可微渲染 + 配置。"""

    def __init__(self, cfg: Config, detector, bundle=None):
        self.cfg = cfg
        self.detector = detector           # 公开 Model-A (白盒, 支持 predict/loss)
        self.device = getattr(cfg.detectors, "device", "cpu")
        self.patch_size = int(cfg.attack.patch_size)
        self.image_size = int(cfg.dataset.image_size)
        self.bundle = bundle
        if bundle is not None:
            self.target_class_ids = bundle.target_class_ids
            self.target_class_names = bundle.target_class_names
        else:
            self.target_class_ids = []
            self.target_class_names = []

    # ---- 可微贴片渲染 (攻击方 EOT 优化用; 公开的是变换族, 隐藏的是评测随机种子) ----
    def sample_params(self, bbox, rng):
        from .patch_renderer import sample_params
        return sample_params(bbox, self.cfg, rng)

    def render_patch_torch(self, image, patch, params):
        from .patch_renderer import render_patch_torch
        return render_patch_torch(image, patch, params, self.image_size)

    def occlusion_patch(self, kind, size, rng):
        from .patch_renderer import occlusion_patch
        return occlusion_patch(kind, size, rng)

    def predict(self, images):
        """便捷: 用公开检测器推理。"""
        return self.detector.predict(images)

    def loss(self, images, targets):
        """便捷: 公开检测器训练损失 (可反传)。"""
        return self.detector.loss(images, targets)


class DefenseEnv:
    """防御方环境: 仅图像尺寸/设备, 无检测器/渲染器。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = getattr(cfg.detectors, "device", "cpu")
        self.image_size = int(cfg.dataset.image_size)


def build_attack_env(cfg, detector, bundle=None) -> AttackEnv:
    return AttackEnv(cfg, detector, bundle)


def build_defense_env(cfg) -> DefenseEnv:
    return DefenseEnv(cfg)
