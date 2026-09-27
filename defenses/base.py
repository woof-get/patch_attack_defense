"""防御方法基类与通用 CNN 二分类训练。

接口 (赛题 §27-§30):
    防御方只接收 image, 返回 patch_probability in [0,1]。
    DefenseModel.predict(image) -> float

BaseDefense.fit(denv, clean_images, patched_images, cfg) -> DefenseModel 实例 (已训练)。
"""
from __future__ import annotations

import numpy as np


class DefenseModel:
    """已训练防御模型基类: 提供 predict(image)->float。子类实现 _forward。"""

    name = "base_defense"

    def __init__(self, device="cpu"):
        self.device = device

    def predict(self, image) -> float:
        import torch
        x = self._preprocess(image)
        with torch.no_grad():
            logit = self._forward(x.unsqueeze(0).to(self.device))
        logit = torch.sigmoid(logit)
        prob = logit.item() if logit.dim() == 0 else logit.view(-1)[0].item()
        return float(np.clip(prob, 0.0, 1.0))

    def _preprocess(self, image):
        raise NotImplementedError

    def _forward(self, x):
        raise NotImplementedError


class BaseDefense:
    """防御方法基类。子类实现 fit() 返回 DefenseModel 实例。"""

    name = "base"
    meta: dict = {}

    def fit(self, denv, clean_images, patched_images, cfg) -> DefenseModel:
        raise NotImplementedError


def as_defense(obj):
    """归一化为 BaseDefense 类 (供 registry 存储, evaluator 实例化)。"""
    if isinstance(obj, type) and issubclass(obj, BaseDefense):
        return obj
    if isinstance(obj, type) and issubclass(obj, DefenseModel):
        # 直接是 DefenseModel 类 (提交形式) -> 包装为 BaseDefense
        class _Wrap(BaseDefense):
            name = obj.__name__

            def fit(self, denv, clean_images, patched_images, cfg):
                return obj()
        return _Wrap
    if callable(obj):
        class _M(DefenseModel):
            def __init__(self, fn, env):
                super().__init__(env.device if env else "cpu")
                self.fn = fn
                self.env = env

            def predict(self, image):
                p = self.fn(self.env, [image])
                return float(np.clip(np.asarray(p).reshape(-1)[0], 0.0, 1.0))

        class _FuncDefense(BaseDefense):
            name = getattr(obj, "name", getattr(obj, "__name__", "func_defense"))

            def __init__(self, **kwargs):
                super().__init__()

            def fit(self, denv, clean_images, patched_images, cfg):
                return _M(obj, denv)
        return _FuncDefense
    raise ValueError(f"无法识别的防御: {obj}")


# ----------------- 通用 CNN 二分类训练 (ResNet18 / MobileNetV3 共用) -----------------

def train_cnn_binary_classifier(name, backbone_builder, clean_images, patched_images,
                                device="cpu", epochs=5, batch_size=32, lr=1e-3,
                                size=224, pretrained=True, seed=42):
    """训练一个 CNN 二分类器 (patched=1, clean=0), 返回 DefenseModel 实例。"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from judge.utils import set_seed, log
    set_seed(seed)

    try:
        net = backbone_builder(pretrained=pretrained)
        log.info(f"[{name}] 骨干网络加载成功 (pretrained={pretrained})")
    except Exception as e:
        log.warning(f"[{name}] 骨干加载失败 ({e}), 回退随机初始化")
        net = backbone_builder(pretrained=False)

    # 替换分类头为 1 logit (sigmoid)
    if hasattr(net, "fc"):
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, 1)
    elif hasattr(net, "classifier"):
        # MobileNetV3: classifier 是 Sequential
        clf = net.classifier
        if hasattr(clf[-1], "in_features"):
            in_f = clf[-1].in_features
            net.classifier[-1] = nn.Linear(in_f, 1)
        else:
            in_f = clf[-1].out_features
            net.classifier = nn.Sequential(clf, nn.Linear(in_f, 1))
    else:
        raise RuntimeError("无法识别的分类头结构")
    net.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    class _DS(Dataset):
        def __init__(s, clean, patched):
            s.x, s.y = [], []
            for im in clean:
                s.x.append(im); s.y.append(0)
            for im in patched:
                s.x.append(im); s.y.append(1)

        def __len__(s):
            return len(s.x)

        def __getitem__(s, i):
            arr = s.x[i].astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            t = torch.from_numpy(np.ascontiguousarray(arr))
            from torchvision.transforms import functional as F
            t = F.resize(t, [size, size])
            return t, s.y[i]

    dl = DataLoader(_DS(clean_images, patched_images), batch_size=batch_size,
                    shuffle=True, num_workers=0, drop_last=False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    net.train()
    log.info(f"[{name}] 训练 {epochs} epochs, {len(clean_images)+len(patched_images)} 样本...")
    for ep in range(epochs):
        tot, n = 0.0, 0
        for x, y in dl:
            x = x.to(device)
            x = (x - mean) / std
            y = y.float().to(device).view(-1, 1)
            logit = net(x)
            loss = bce(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * x.shape[0]; n += x.shape[0]
        log.info(f"[{name}] epoch {ep+1}/{epochs}  loss={tot/max(1,n):.4f}")
    net.eval()

    class _CNNDefense(DefenseModel):
        pass

    m = _CNNDefense(device=device)
    m.name = name
    m.net = net
    m.size = size
    m.mean = mean
    m.std = std
    m._preprocess_fn = None

    def _preprocess(self, image):
        arr = np.asarray(image).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        t = torch.from_numpy(np.ascontiguousarray(arr))
        from torchvision.transforms import functional as F
        return F.resize(t, [self.size, self.size])

    def _forward(self, x):
        x = (x - self.mean) / self.std
        return self.net(x).view(-1)[0]

    # 绑定方法
    import types
    m._preprocess = types.MethodType(_preprocess, m)
    m._forward = types.MethodType(_forward, m)
    return m
