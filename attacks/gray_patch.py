"""自然遮挡贴片攻击 (gray/black/white), 用于对照"纯遮挡"基线。

注意: 裁判端 §19 的自然遮挡基线用相同参数渲染 gray/black/white/noise/texture,
因此此类攻击的 AttackGain 预期接近 0 (仅遮挡, 无对抗增益), 正好验证
AttackGain 公式能正确扣减遮挡影响。
"""
import numpy as np

from .base import BaseAttack


class GrayPatchAttack(BaseAttack):
    name = "gray_patch"
    meta = {"type": "occlusion_baseline", "reference": "natural_occlusion"}

    def __init__(self, color="gray", seed=42):
        self.color = color
        self.seed = seed

    def __call__(self, env, task):
        ps = int(task["patch_size"])
        if self.color == "gray":
            p = np.full((3, ps, ps), 0.5, dtype=np.float32)
        elif self.color == "black":
            p = np.zeros((3, ps, ps), dtype=np.float32)
        elif self.color == "white":
            p = np.ones((3, ps, ps), dtype=np.float32)
        else:
            rng = np.random.default_rng(self.seed)
            p = rng.random((3, ps, ps), dtype=np.float32)
        return {"patch": p}
