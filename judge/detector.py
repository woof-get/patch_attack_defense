"""统一目标检测器封装 (Model-A / Model-B)。

- predict(images): 推理, 返回统一格式 [{boxes, labels, scores}, ...]
- loss(images, targets): 训练模式, 返回检测损失 dict (供梯度攻击反传)
- 冻结权重, 仅放行对输入图像的梯度。

默认用 torchvision 预训练 COCO 检测器 (替代赛题"建议"的 YOLO, 因本环境
无 ultralytics 且 CPU 下 torchvision 更稳; 经 Detector 抽象可后续插入 YOLO)。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .utils import log

# COCO 类别 id -> 名称 (torchvision 预训练检测器输出的 label 即 COCO id)
COCO_CATEGORY_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
    51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
    56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}
NAME_TO_COCO_ID = {v: k for k, v in COCO_CATEGORY_NAMES.items()}


class Detector:
    """检测器抽象基类。子类实现 _build_model()。"""

    name = "base"
    num_classes = 91  # COCO 含背景

    def __init__(self, device: str = "cpu", score_thresh: float = 0.25,
                 nms_iou: float = 0.45, input_size: int = 320):
        self.device = device
        self.score_thresh = score_thresh
        self.nms_iou = nms_iou
        self.input_size = input_size
        self._model = None

    # ---- 子类实现 ----
    def _build_model(self):
        raise NotImplementedError

    @property
    def model(self):
        if self._model is None:
            log.info(f"[detector] 加载 {self.name} (首次需下载权重, 请稍候)...")
            self._model = self._build_model()
            self._configure_thresholds()
            self._freeze()
            self._model.to(self.device)
        return self._model

    def _configure_thresholds(self):
        m = self._model
        # Faster R-CNN / RetinaNet 族: roi_heads 上有 score_thresh / nms_thresh
        if hasattr(m, "roi_heads"):
            if hasattr(m.roi_heads, "score_thresh"):
                m.roi_heads.score_thresh = self.score_thresh
            if hasattr(m.roi_heads, "nms_thresh"):
                m.roi_heads.nms_thresh = self.nms_iou

    def _freeze(self):
        for p in self._model.parameters():
            p.requires_grad_(False)

    # ---- 统一接口 ----
    def _to_tensor(self, images):
        """images: ndarray(HWC uint8|float) | CHW | tensor | list[...] -> list[CHW float tensor]。"""
        import torch
        if not isinstance(images, (list, tuple)):
            images = [images]
        out = []
        for img in images:
            if isinstance(img, np.ndarray):
                arr = img.astype(np.float32)
                if arr.max() > 1.5:
                    arr = arr / 255.0
                if arr.ndim == 3 and arr.shape[-1] in (1, 3):  # HWC -> CHW
                    arr = np.transpose(arr, (2, 0, 1))
                t = torch.from_numpy(np.ascontiguousarray(arr))
            else:  # tensor
                t = img.float()
                if t.ndim == 3 and t.shape[-1] in (1, 3):
                    t = t.permute(2, 0, 1)
            out.append(t.to(self.device))
        return out

    @staticmethod
    def _targets_to_tensor(targets, device):
        """targets: list[dict{boxes(HWC? no, xyxy), labels}] ndarray -> torch 格式。"""
        import torch
        out = []
        for t in targets:
            boxes = t["boxes"]
            if isinstance(boxes, np.ndarray):
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
            else:
                boxes = boxes.clone().float()
            labels = t["labels"]
            if isinstance(labels, np.ndarray):
                labels = torch.as_tensor(labels, dtype=torch.long)
            else:
                labels = labels.clone().long()
            out.append({"boxes": boxes.to(device), "labels": labels.to(device)})
        return out

    def predict(self, images):
        """推理。返回 list[dict{boxes(N,4)xyxy, labels(N,), scores(N,)}] (numpy)。"""
        import torch
        m = self.model
        m.eval()
        tensors = self._to_tensor(images)
        with torch.no_grad():
            outs = m(tensors)
        results = []
        for o in outs:
            boxes = o["boxes"].cpu().numpy().astype(np.float32)
            labels = o["labels"].cpu().numpy().astype(np.int64)
            scores = o["scores"].cpu().numpy().astype(np.float32)
            keep = scores >= self.score_thresh
            results.append({
                "boxes": boxes[keep] if keep.any() else boxes[:0].reshape(0, 4),
                "labels": labels[keep] if keep.any() else labels[:0],
                "scores": scores[keep] if keep.any() else scores[:0],
            })
        return results

    def loss(self, images, targets):
        """训练模式前向, 返回 (带梯度的输入张量列表, 检测损失 dict)。

        用于梯度贴片攻击: 攻击方最大化检测损失以隐藏目标。
        不 detach 输入 -> 保留 patch -> 渲染图 -> 检测器 的计算图。
        """
        m = self.model
        m.train()
        tensors = self._to_tensor(images)
        # 若输入已 require_grad (来自贴片渲染), 保留计算图; 否则建叶节点
        tensors = [t if t.requires_grad else t.detach().requires_grad_(True) for t in tensors]
        tgt = self._targets_to_tensor(targets, self.device)
        losses = m(tensors, tgt)
        return tensors, losses

    def hiding_loss(self, images, targets):
        """隐藏目标损失 (默认实现: 最大化 objectness/分类训练损失)。

        返回 (带梯度的输入张量列表, 标量 loss)。攻击方 *最小化* 该 loss 以隐藏目标。
        子类可重写为更直接/不饱和的目标 (如直接最小化 objectness logit)。
        """
        tensors, losses = self.loss(images, targets)
        hide = sum(v for k, v in losses.items() if "box" not in k.lower())
        return tensors, -hide  # 最小化 -hide = 最大化 hide  # 返回带梯度的输入张量 + 损失字典

    def class_name(self, coco_id: int) -> str:
        return COCO_CATEGORY_NAMES.get(int(coco_id), f"class_{coco_id}")


# ----------------- torchvision 具体检测器 -----------------

class TorchvisionDetector(Detector):
    """通用 torchvision 检测器包装。"""

    def __init__(self, name: str, builder, device="cpu", score_thresh=0.25,
                 nms_iou=0.45, input_size=320, weights="DEFAULT"):
        super().__init__(device, score_thresh, nms_iou, input_size)
        self.name = name
        self._builder = builder
        self._weights = weights

    def _build_model(self):
        return self._builder(weights=self._weights)


def _build_mobilenet_frcnn(weights="DEFAULT"):
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    )
    w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if weights == "DEFAULT" else weights
    return fasterrcnn_mobilenet_v3_large_fpn(weights=w)


def _build_resnet50_frcnn(weights="DEFAULT"):
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if weights == "DEFAULT" else weights
    return fasterrcnn_resnet50_fpn(weights=w)


def _build_retinanet(weights="DEFAULT"):
    from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
    w = RetinaNet_ResNet50_FPN_Weights.DEFAULT if weights == "DEFAULT" else weights
    return retinanet_resnet50_fpn(weights=w)


_BUILDERS = {
    "fasterrcnn_mobilenet_v3_large_fpn": _build_mobilenet_frcnn,
    "fasterrcnn_resnet50_fpn": _build_resnet50_frcnn,
    "retinanet_resnet50_fpn": _build_retinanet,
}


def build_detector(cfg, role: str = "model_a") -> Optional[Detector]:
    """根据 config.detectors 构建检测器。role: 'model_a' | 'model_b'。"""
    dcfg = getattr(cfg.detectors, role, None)
    if dcfg is None:
        return None
    if role == "model_b" and not getattr(dcfg, "enabled", False):
        return None
    dtype = dcfg.type
    if dtype not in _BUILDERS:
        raise ValueError(f"不支持的检测器类型: {dtype}, 可选: {list(_BUILDERS)}")
    det = TorchvisionDetector(
        name=role,
        builder=_BUILDERS[dtype],
        device=getattr(cfg.detectors, "device", "cpu"),
        score_thresh=getattr(dcfg, "score_thresh", 0.25),
        nms_iou=getattr(dcfg, "nms_iou", 0.45),
        input_size=getattr(cfg.dataset, "image_size", 320),
        weights=getattr(dcfg, "weights", "DEFAULT"),
    )
    return det


def build_detectors(cfg) -> List[Detector]:
    """构建所有启用的检测器 (至少 Model-A)。"""
    dets = []
    a = build_detector(cfg, "model_a")
    if a is not None:
        dets.append(a)
    b = build_detector(cfg, "model_b")
    if b is not None:
        dets.append(b)
    return dets
