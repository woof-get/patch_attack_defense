"""A0-2 棋盘纹理贴片 (非优化高频纹理基线, 赛题 §45)。"""
from .base import BaseAttack


class CheckerboardAttack(BaseAttack):
    name = "checkerboard"
    meta = {"type": "baseline", "reference": "A0-2"}

    def __init__(self, cells=8, seed=42):
        self.cells = cells
        self.seed = seed

    def __call__(self, env, task):
        import numpy as np
        ps = int(task["patch_size"])
        rng = np.random.default_rng(self.seed)
        p = np.zeros((3, ps, ps), dtype=np.float32)
        cell = max(2, ps // self.cells)
        # 随机两色
        c0 = rng.random(3)
        c1 = rng.random(3)
        for i in range(0, ps, cell):
            for j in range(0, ps, cell):
                c = c0 if ((i // cell) + (j // cell)) % 2 == 0 else c1
                p[:, i:i + cell, j:j + cell] = c[:, None, None]
        return {"patch": p}
