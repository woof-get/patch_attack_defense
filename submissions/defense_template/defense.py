"""选手防御提交模板 (赛题 §27-§30/§56/§58)。

接口 (二选一):
    A) class DefenseModel:
           def __init__(self, model_path): ...
           def predict(self, image) -> float   # patch_probability in [0,1]
    B) def defend(env, images) -> ndarray[N]   # 概率数组

约束:
    - 只接收 image, 不获框/类别/贴片/检测预测
    - 不得修改/替换检测器, 不得返回人为规则标签
    - 输出 [0,1] 概率 (1=含攻击贴片, 0=正常)

本模板实现一个基于高频能量比的简单防御 (无训练, 仅验证接口)。
真实防御可参考 defenses/resnet18_classifier.py / frequency_classifier.py。
"""
import numpy as np


class DefenseModel:
    """提交时可在 model/ 下放训练好的权重, __init__ 加载。"""

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.thresh = 0.05  # 高频能量比阈值 (演示用)

    def predict(self, image) -> float:
        """image: HWC uint8 RGB -> float [0,1]。"""
        from PIL import Image
        g = np.asarray(Image.fromarray(image.astype(np.uint8)).convert("L").resize((64, 64)), dtype=np.float32) / 255.0
        F = np.fft.fft2(g)
        mag = np.abs(np.fft.fftshift(F))
        total = (mag ** 2).sum() + 1e-6
        cy, cx = 32, 32
        yy, xx = np.mgrid[0:64, 0:64]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        high_ratio = float((mag[r > 16] ** 2).sum() / total)
        # 简单线性映射到概率
        prob = max(0.0, min(1.0, (high_ratio - self.thresh) * 8.0))
        return float(prob)


# 或提交函数形式:
# def defend(env, images):
#     import numpy as np
#     out = []
#     for img in images:
#         out.append(...)
#     return np.asarray(out, dtype=np.float32)
