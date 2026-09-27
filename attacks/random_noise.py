"""A0-1 随机噪声贴片 (最低攻击基线, 赛题 §45 / §53)。"""
from .base import BaseAttack


class RandomNoiseAttack(BaseAttack):
    name = "random_noise"
    meta = {"type": "baseline", "reference": "A0-1"}

    def __init__(self, seed=42):
        self.seed = seed

    def __call__(self, env, task):
        import numpy as np
        ps = int(task["patch_size"])
        rng = np.random.default_rng(self.seed)
        patch = rng.random((3, ps, ps), dtype=np.float32)
        return {"patch": patch}
