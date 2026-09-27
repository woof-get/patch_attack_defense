"""选手攻击提交模板 (赛题 §15-§18/§53/§57)。

接口:
    def attack(env, task) -> {"patch": ndarray (3, patch_size, patch_size) float [0,1]}

task = {
    "images":               list[HWC uint8]  攻击优化训练图像
    "annotations":          list[list[{bbox, category_id}]]
    "target_class":         int   目标 COCO 类别 id
    "patch_size":           int
    "max_patch_area_ratio": float
    "detector":             env.detector   公开 Model-A (predict/loss, 白盒可反传)
}

约束:
    - 只返回 patch, 不返回改图/位置/缩放/旋转 (裁判统一渲染)
    - patch.shape == (3, patch_size, patch_size), 像素 [0,1], 有限
    - 同一轮所有测试样本使用同一张贴片 (Universal)
    - 不得修改检测器/标签/隐藏测试图

本模板实现一个最简单的随机噪声贴片 (无真实攻击力, 仅验证接口)。
真实攻击可参考 attacks/gradient_patch.py (DPatch + EOT + 梯度优化)。
"""
import numpy as np


def attack(env, task):
    patch_size = int(task["patch_size"])
    rng = np.random.default_rng(42)
    patch = rng.random((3, patch_size, patch_size), dtype=np.float32)
    return {"patch": patch}


# 也可以提交 BaseAttack 子类 (二选一, 裁判均识别)
# from attacks.base import BaseAttack
# class MyAttack(BaseAttack):
#     name = "my_attack"
#     def __call__(self, env, task):
#         ...
#         return {"patch": patch}
