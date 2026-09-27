"""频域特征二分类防御 (无大权重依赖, CPU 极快)。

思路: 对抗贴片常含高频/非自然纹理, 提取 FFT 幅度谱统计 + 高频能量比作为特征,
用逻辑回归二分类。不依赖 ImageNet 预训练权重, 适合离线快速验证。
"""
import numpy as np

from .base import BaseDefense, DefenseModel


def _extract_features(image, size=64):
    """image: HWC uint8 -> 1D 特征向量。"""
    from PIL import Image
    img = Image.fromarray(image.astype(np.uint8)).convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    F = np.fft.fft2(a)
    mag = np.abs(np.fft.fftshift(F)) + 1e-3
    log_mag = np.log(mag)
    # 径向平均 (频域能量分布)
    cy, cx = size // 2, size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = r.max()
    radial = np.zeros(rmax + 1, dtype=np.float32)
    cnt = np.zeros(rmax + 1, dtype=np.float32)
    np.add.at(radial, r, log_mag)
    np.add.at(cnt, r, 1.0)
    radial = radial / np.maximum(cnt, 1.0)
    # 高频/低频能量比
    total = (log_mag ** 2).sum() + 1e-6
    low_mask = r <= (size // 8)
    high_ratio = float((log_mag[~low_mask] ** 2).sum() / total)
    # 颜色统计 (贴片常改变局部颜色分布)
    col = image.astype(np.float32).reshape(-1, 3)
    col_mean = col.mean(0)
    col_std = col.std(0)
    feat = np.concatenate([radial, [high_ratio], col_mean, col_std]).astype(np.float32)
    return feat


class FrequencyDefense(BaseDefense):
    name = "frequency_classifier"
    meta = {"type": "frequency", "reference": "custom"}

    def fit(self, denv, clean_images, patched_images, cfg):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from judge.utils import log
        tcfg = getattr(cfg, "training", None)
        dcfg = getattr(tcfg, "defense", None) if tcfg else None
        max_per = int(getattr(dcfg, "batch_size", 32)) * 20  # 限制特征提取数量加速

        def feats(imgs):
            xs = []
            for im in imgs[:max_per]:
                xs.append(_extract_features(im))
            return np.stack(xs)

        Xc, Xp = feats(clean_images), feats(patched_images)
        X = np.concatenate([Xc, Xp], axis=0)
        y = np.concatenate([np.zeros(len(Xc)), np.ones(len(Xp))])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xs, y)
        train_acc = float(clf.score(Xs, y))
        log.info(f"[{self.name}] 频域分类器训练完成, train_acc={train_acc:.3f}, feat_dim={X.shape[1]}")

        m = DefenseModel(device="cpu")
        m.name = self.name
        m.scaler = scaler
        m.clf = clf

        def _preprocess(self, image):
            return _extract_features(image)

        def _forward(self, x):
            # x: (feat_dim,) -> 标量 logit
            import numpy as np
            xs = self.scaler.transform(x.cpu().numpy().reshape(1, -1))
            return float(self.clf.decision_function(xs)[0])

        import torch, types
        # _forward 返回 numpy float; predict 期望 torch.sigmoid(logit)
        # 改写 predict 以兼容 sklearn (直接返回概率)
        def predict(self, image):
            f = _extract_features(image).reshape(1, -1)
            xs = self.scaler.transform(f)
            prob = self.clf.predict_proba(xs)[0, 1]
            return float(np.clip(prob, 0.0, 1.0))
        m.predict = types.MethodType(predict, m)
        return m
