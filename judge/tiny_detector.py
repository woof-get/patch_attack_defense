"""轻量可微单阶段检测器 (TinyDetector), 仅用于合成数据验证模式。

为什么需要它: 预训练 COCO 检测器无法识别合成彩色形状, 而 COCO 官方数据集在
离线/慢速网络下无法下载。为保证"攻防对抗后端"在无网络时也能端到端验证
攻击(需可微以支持梯度贴片)/防御算法, 这里训练一个紧凑的可微检测器作为
合成模式的 Model-A。真实 COCO 模式仍用 torchvision 预训练检测器(见 detector.py)。

设计: 单阶段, 网格 20x20 (stride16@320), 每格预测 objectness + K 类 + box。
FCOS 风格 per-cell 回归, sigmoid 归一化 [0,1]*H 的 xyxy。端到端可微。
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from .detector import Detector
from .utils import log


class _TinyNet:
    """torch 模型 + 训练/推理逻辑。延迟 import torch。"""

    def __init__(self, num_classes: int, image_size: int = 320, device: str = "cpu"):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.num_classes = num_classes
        self.image_size = image_size
        self.stride = 16
        self.grid = image_size // self.stride
        self.device = device

        def conv_bn(ci, co, k=3, s=2, p=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, s, p, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True))

        c = 24
        self.backbone = nn.Sequential(
            conv_bn(3, c), conv_bn(c, c * 2), conv_bn(c * 2, c * 4),
            conv_bn(c * 4, c * 4)).to(device)
        self.head_obj = nn.Conv2d(c * 4, 1, 1).to(device)
        self.head_cls = nn.Conv2d(c * 4, num_classes, 1).to(device)
        self.head_box = nn.Conv2d(c * 4, 4, 1).to(device)
        self.target_classes_local = {}  # coco id -> local idx

    def parameters(self):
        for m in [self.backbone, self.head_obj, self.head_cls, self.head_box]:
            yield from m.parameters()

    def state_dict(self):
        return {"backbone": self.backbone.state_dict(),
                "head_obj": self.head_obj.state_dict(),
                "head_cls": self.head_cls.state_dict(),
                "head_box": self.head_box.state_dict(),
                "num_classes": self.num_classes,
                "image_size": self.image_size}

    def load_state_dict(self, sd):
        self.backbone.load_state_dict(sd["backbone"])
        self.head_obj.load_state_dict(sd["head_obj"])
        self.head_cls.load_state_dict(sd["head_cls"])
        self.head_box.load_state_dict(sd["head_box"])

    def forward(self, x):
        feat = self.backbone(x)
        obj = self.head_obj(feat)          # (B,1,G,G)
        cls = self.head_cls(feat)          # (B,K,G,G)
        raw = self.head_box(feat)          # (B,4,G,G) = (l,t,r,b) 原始值
        dist = torch.exp(raw.clamp(-6.0, 6.0))  # 正距离
        G = self.grid
        coords = (torch.arange(G, device=x.device).float() + 0.5) * self.stride
        cx_grid = coords.view(1, G).expand(G, G)  # 列 x 坐标 (随 j 变)
        cy_grid = coords.view(G, 1).expand(G, G)  # 行 y 坐标 (随 i 变)
        l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
        x1 = cx_grid.unsqueeze(0) - l
        y1 = cy_grid.unsqueeze(0) - t
        x2 = cx_grid.unsqueeze(0) + r
        y2 = cy_grid.unsqueeze(0) + b
        box = torch.stack([x1, y1, x2, y2], dim=1)  # (B,4,G,G) xyxy
        return obj, cls, box

    def _cell_centers(self, device):
        G = self.grid
        coords = (torch.arange(G, device=device).float() + 0.5) * self.stride
        cx = coords.view(1, G).expand(G, G)
        cy = coords.view(G, 1).expand(G, G)
        return cx, cy

    def compute_loss(self, x, targets):
        """targets: list[dict{boxes(xyxy), labels(long, coco id)}]。"""
        torch = self.torch
        B = x.shape[0]
        obj, cls, box = self.forward(x)
        G = self.grid
        device = x.device
        H = self.image_size
        target_classes_local = self.target_classes_local  # coco id -> local idx

        obj_tgt = torch.zeros(B, 1, G, G, device=device)
        cls_tgt = torch.zeros(B, self.num_classes, G, G, device=device)
        box_tgt = torch.zeros(B, 4, G, G, device=device)
        pos_mask = torch.zeros(B, 1, G, G, device=device)

        # 网格单元中心坐标 (像素)
        coords = (torch.arange(G, device=device).float() + 0.5) * self.stride  # (G,)

        for b, t in enumerate(targets):
            boxes = t["boxes"]
            labels = t["labels"]
            if len(boxes) == 0:
                continue
            for bx, lb in zip(boxes, labels):
                x1, y1, x2, y2 = [float(v) for v in bx]
                # FCOS 风格: 中心落在 GT 框内的所有单元均为正样本
                gx_in = (coords >= x1) & (coords <= x2)
                gy_in = (coords >= y1) & (coords <= y2)
                gi_idx = torch.nonzero(gy_in, as_tuple=True)[0]
                gj_idx = torch.nonzero(gx_in, as_tuple=True)[0]
                if len(gi_idx) == 0 or len(gj_idx) == 0:
                    # 退化: 用中心单元
                    gi = int(min(max(((y1 + y2) / 2) / self.stride, 0), G - 1))
                    gj = int(min(max(((x1 + x2) / 2) / self.stride, 0), G - 1))
                    obj_tgt[b, 0, gi, gj] = 1.0
                    pos_mask[b, 0, gi, gj] = 1.0
                    box_tgt[b, :, gi, gj] = torch.tensor([x1, y1, x2, y2], device=device)
                    if int(lb) in target_classes_local:
                        cls_tgt[b, target_classes_local[int(lb)], gi, gj] = 1.0
                    continue
                for gi in gi_idx:
                    for gj in gj_idx:
                        obj_tgt[b, 0, gi, gj] = 1.0
                        pos_mask[b, 0, gi, gj] = 1.0
                        box_tgt[b, :, gi, gj] = torch.tensor([x1, y1, x2, y2], device=device)
                        if int(lb) in target_classes_local:
                            cls_tgt[b, target_classes_local[int(lb)], gi, gj] = 1.0

        # objectness: BCE with pos_weight 缓解正负样本极度不平衡
        pos_count = pos_mask.sum().clamp(min=1.0)
        neg_count = (B * G * G - pos_count).clamp(min=1.0)
        pw = (neg_count / pos_count).clamp(max=50.0)
        obj_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            obj, obj_tgt, reduction="mean", pos_weight=pw)
        # classification only on positive cells
        pos = pos_mask.expand_as(cls_tgt)
        cls_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            cls, cls_tgt, reduction="none") * pos).sum() / (pos.sum() + 1e-6)
        # box L1 (归一化到 [0,1]) on positive cells
        box_norm = box / H
        box_tgt_norm = box_tgt / H
        box_loss = (torch.nn.functional.l1_loss(box_norm, box_tgt_norm, reduction="none")
                    * pos[:, 0:1]).sum() / (pos.sum() + 1e-6)
        return {"loss_obj": obj_loss, "loss_cls": cls_loss * 2.0, "loss_box": box_loss * 5.0}


class TinyDetector(Detector):
    """合成模式用的可微检测器 (Detector 接口)。"""

    name = "tiny_detector_synth"

    def __init__(self, target_class_ids: List[int], device="cpu",
                 score_thresh=0.25, nms_iou=0.45, input_size=320, weights_path=None):
        super().__init__(device, score_thresh, nms_iou, input_size)
        self.target_class_ids = list(target_class_ids)
        self.coco_to_local = {cid: i for i, cid in enumerate(self.target_class_ids)}
        self.weights_path = weights_path

    @property
    def model(self):
        if self._model is None:
            self._model = _TinyNet(num_classes=len(self.target_class_ids),
                                   image_size=self.input_size, device=self.device)
            self._model.target_classes_local = self.coco_to_local
            if self.weights_path:
                import torch
                sd = torch.load(self.weights_path, map_location=self.device)
                self._model.load_state_dict(sd)
                log.info(f"[detector] TinyDetector 已加载权重 {self.weights_path}")
        return self._model

    def predict(self, images):
        import torch
        from torchvision.ops import nms
        m = self.model
        m.backbone.eval()
        tensors = self._to_tensor(images)
        results = []
        with torch.no_grad():
            for x in tensors:
                x = x.unsqueeze(0)
                obj, cls, box = m.forward(x)
                obj_p = torch.sigmoid(obj[0, 0])          # (G,G)
                cls_logits = cls[0]                        # (K,G,G)
                box_xyxy = box[0]                          # (4,G,G)
                # 取所有 objectness 超阈值的格点
                mask = obj_p >= self.score_thresh
                if mask.sum() == 0:
                    results.append({"boxes": np.zeros((0, 4), np.float32),
                                    "labels": np.zeros((0,), np.int64),
                                    "scores": np.zeros((0,), np.float32)})
                    continue
                idx = mask.flatten().nonzero(as_tuple=True)[0]
                gi = idx // m.grid
                gj = idx % m.grid
                scores = obj_p[gi, gj]
                boxes = box_xyxy[:, gi, gj].T  # (N,4)
                labels_local = cls_logits[:, gi, gj].argmax(0)
                labels = torch.tensor([self.target_class_ids[int(l)] for l in labels_local],
                                      device=x.device)
                keep = nms(boxes, scores, self.nms_iou)
                results.append({
                    "boxes": boxes[keep].cpu().numpy().astype(np.float32),
                    "labels": labels[keep].cpu().numpy().astype(np.int64),
                    "scores": scores[keep].cpu().numpy().astype(np.float32),
                })
        return results

    def loss(self, images, targets):
        m = self.model
        m.backbone.train()
        tensors = self._to_tensor(images)
        # 保留 patch -> 渲染图 -> 检测器 的计算图
        tensors = [t if t.requires_grad else t.detach().requires_grad_(True) for t in tensors]
        tgt = self._targets_to_tensor(targets, self.device)
        # 组 batch (合成图同尺寸可 stack)
        import torch
        x = torch.stack(tensors)
        losses = m.compute_loss(x, tgt)
        return tensors, losses

    def hiding_loss(self, images, targets):
        """直接隐藏目标: 最小化目标框内 objectness logit 均值 + 破坏框回归。

        直接对 logit 优化 (非饱和, 梯度恒定), 比 BCE 训练损失更有效。
        攻击方 *最小化* 返回的 loss。
        """
        m = self.model
        m.backbone.eval()  # 稳定 BatchNorm, 减少梯度噪声
        tensors = self._to_tensor(images)
        tensors = [t if t.requires_grad else t.detach().requires_grad_(True) for t in tensors]
        x = torch.stack(tensors)
        obj, cls, box = m.forward(x)  # obj:(B,1,G,G) logits, box:(B,4,G,G) xyxy  # noqa: F841
        G = m.grid
        device = x.device
        coords = (torch.arange(G, device=device).float() + 0.5) * m.stride
        obj_term = x.new_zeros(())
        box_disrupt = x.new_zeros(())
        n = 0
        for b, t in enumerate(targets):
            for bx in t["boxes"]:
                x1, y1, x2, y2 = [float(v) for v in bx]
                gx_in = (coords >= x1) & (coords <= x2)
                gy_in = (coords >= y1) & (coords <= y2)
                if gx_in.sum() == 0 or gy_in.sum() == 0:
                    continue
                obj_region = obj[b, 0][gy_in][:, gx_in]          # (ny,nx) logits
                box_region = box[b, :, gy_in][:, :, gx_in]        # (4,ny,nx) xyxy
                # logsumexp 近似 max: 最小化它会把"最自信的格点"压下去 (平滑, 全格点有梯度)
                obj_term = obj_term + torch.logsumexp(obj_region.reshape(-1), dim=0)
                gt = torch.tensor([x1, y1, x2, y2], device=device).view(4, 1, 1)
                box_disrupt = box_disrupt - (box_region - gt).abs().mean()
                n += 1
        loss = (obj_term + 0.3 * box_disrupt) / max(1, n)
        return tensors, loss  # 最小化 -> 压低最自信格点 + 扰乱框


def train_tiny_detector(images, annotations, target_class_ids, image_size=320,
                        epochs=15, lr=1e-3, batch_size=8, device="cpu",
                        save_path=None, seed=42):
    """在合成数据上训练 TinyDetector。images: list[HWC uint8]; annotations: list[list[{bbox,cid}]]。"""
    from .utils import set_seed
    import torch
    from torch.utils.data import DataLoader, Dataset
    set_seed(seed)
    net = _TinyNet(num_classes=len(target_class_ids), image_size=image_size, device=device)
    net.target_classes_local = {cid: i for i, cid in enumerate(target_class_ids)}

    class _DS(Dataset):
        def __init__(s, imgs, anns):
            s.imgs, s.anns = imgs, anns

        def __len__(s):
            return len(s.imgs)

        def __getitem__(s, i):
            arr = s.imgs[i].astype(np.float32) / 255.0
            if arr.shape[-1] in (1, 3):
                arr = np.transpose(arr, (2, 0, 1))
            t = torch.from_numpy(np.ascontiguousarray(arr))
            boxes = torch.tensor([a["bbox"] for a in s.anns[i]], dtype=torch.float32) if s.anns[i] else torch.zeros((0, 4))
            labels = torch.tensor([a["category_id"] for a in s.anns[i]], dtype=torch.long) if s.anns[i] else torch.zeros((0,), dtype=torch.long)
            return t, {"boxes": boxes, "labels": labels}

    dl = DataLoader(_DS(images, annotations), batch_size=batch_size, shuffle=True,
                    num_workers=0, drop_last=False,
                    collate_fn=lambda batch: (torch.stack([b[0] for b in batch]),
                                              [b[1] for b in batch]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    log.info(f"[tiny_detector] 训练 {epochs} epochs, {len(images)} 样本, {len(target_class_ids)} 类...")
    for ep in range(epochs):
        tot = 0.0
        for x, tgt in dl:
            x = x.to(device)
            tgt = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in tgt]
            losses = net.compute_loss(x, tgt)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * x.shape[0]
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            log.info(f"[tiny_detector] epoch {ep+1}/{epochs}  loss={tot/len(images):.4f}")
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(net.state_dict(), save_path)
        log.info(f"[tiny_detector] 权重已保存 {save_path}")
    return net
