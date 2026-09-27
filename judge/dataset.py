"""数据集加载: COCO(官方) 与 合成 数据统一为同一格式, 含类别筛选/划分/letterbox 缩放。

Sample 结构 (统一):
    {
      "image":      np.ndarray HxWx3 uint8 (已缩放到 image_size, letterbox)
      "image_id":   int | str
      "annotations":[{"bbox": [x1,y1,x2,y2] (image_size 坐标), "category_id": int, "category_name": str}, ...]
      "source":     str  # 原始文件路径(调试)
    }

划分: attack_train / attack_val / defense_train_clean / defense_val_clean / hidden_test_clean
防御训练/验证的 "patched" 一半由 evaluator 用基线攻击池渲染生成(此处只提供 clean)。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import Config, resolve_path
from .detector import NAME_TO_COCO_ID, COCO_CATEGORY_NAMES
from .utils import log, load_image


class Sample(dict):
    """属性访问的样本字典。"""
    @property
    def image(self):
        return self["image"]

    @property
    def annotations(self):
        return self["annotations"]

    @property
    def image_id(self):
        return self["image_id"]


class DatasetBundle:
    def __init__(self):
        self.attack_train: List[Sample] = []
        self.attack_val: List[Sample] = []
        self.defense_train_clean: List[Sample] = []
        self.defense_val_clean: List[Sample] = []
        self.hidden_test_clean: List[Sample] = []
        self.target_class_ids: List[int] = []   # COCO category ids
        self.target_class_names: List[str] = []
        self.source: str = "unknown"
        self.image_size: int = 320

    def __repr__(self):
        return (f"DatasetBundle(source={self.source}, target={self.target_class_names}, "
                f"atk_train={len(self.attack_train)}, atk_val={len(self.attack_val)}, "
                f"def_train={len(self.defense_train_clean)}, def_val={len(self.defense_val_clean)}, "
                f"hidden={len(self.hidden_test_clean)})")


# ----------------- 图像缩放 (letterbox, 保比例, 灰边填充) -----------------

def letterbox(image: np.ndarray, size: int, pad_value: int = 114) -> tuple:
    """image: HxWx3 uint8 -> (out HxWx3 uint8, scale, pad_x, pad_y)。保比例, 灰边填充。"""
    from PIL import Image
    h, w = image.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = np.array(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, s, pad_x, pad_y


def _scale_bbox_xyxy(bbox, s, pad_x, pad_y):
    x1, y1, x2, y2 = bbox
    return [x1 * s + pad_x, y1 * s + pad_y, x2 * s + pad_x, y2 * s + pad_y]


# ----------------- COCO JSON 解析(自实现, 不依赖 pycocotools) -----------------

def _parse_coco_json(ann_path: str) -> Dict:
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    imgs = {im["id"]: im for im in data["images"]}
    cats = {c["id"]: c for c in data["categories"]}
    by_img = {im_id: [] for im_id in imgs}
    for a in data["annotations"]:
        if a["image_id"] in by_img:
            by_img[a["image_id"]].append(a)
    return {"images": imgs, "categories": cats, "ann_by_image": by_img}


def _split_key(image_id, n_buckets: int = 100) -> int:
    """确定性把 image_id 映射到 [0, n_buckets), 用于稳定划分。"""
    h = hashlib.md5(str(image_id).encode()).hexdigest()
    return int(h[:8], 16) % n_buckets


# ----------------- 主加载入口 -----------------

def load_dataset(cfg: Config) -> DatasetBundle:
    """根据 cfg.dataset.source 加载数据集。auto: 优先 coco, 缺失用 synthetic。"""
    src = (cfg.dataset.source or "auto").lower()
    image_size = int(getattr(cfg.dataset, "image_size", 320))
    target_names = list(cfg.dataset.target_classes or ["person"])
    target_ids = [NAME_TO_COCO_ID[n] for n in target_names if n in NAME_TO_COCO_ID]
    if not target_ids:
        raise ValueError(f"无法识别的目标类别: {target_names}")

    bundle = DatasetBundle()
    bundle.image_size = image_size
    bundle.target_class_ids = target_ids
    bundle.target_class_names = target_names

    coco_root = resolve_path(cfg, cfg.dataset.root)
    syn_root = resolve_path(cfg, cfg.dataset.synthetic_root)

    if src == "auto":
        if (coco_root / "annotations" / "instances_val2017.json").exists() and \
           (coco_root / "val2017").exists():
            src = "coco"
        elif syn_root.exists():
            src = "synthetic"
        else:
            src = "synthetic"  # 后续会触发生成

    if src == "coco":
        _load_coco(bundle, coco_root, cfg)
    else:
        _load_synthetic(bundle, syn_root, cfg)

    log.info(f"[dataset] {bundle}")
    return bundle


def _make_sample(image: np.ndarray, image_id, anns_raw, s, pad_x, pad_y, target_ids, source):
    """构建一个 Sample, 过滤目标类别 + 缩放 bbox。image 已 letterbox。"""
    kept = []
    for a in anns_raw:
        cid = a["category_id"]
        if cid not in target_ids:
            continue
        bx, by, bw, bh = a["bbox"]  # COCO xywh
        bbox = [bx, by, bx + bw, by + bh]
        bbox = _scale_bbox_xyxy(bbox, s, pad_x, pad_y)
        # 跳过过小目标
        if (bbox[2] - bbox[0]) < 4 or (bbox[3] - bbox[1]) < 4:
            continue
        kept.append({
            "bbox": [float(v) for v in bbox],
            "category_id": int(cid),
            "category_name": COCO_CATEGORY_NAMES.get(int(cid), str(cid)),
        })
    return Sample({
        "image": image,
        "image_id": image_id,
        "annotations": kept,
        "source": source,
    })


def _load_coco(bundle: DatasetBundle, coco_root: Path, cfg: Config):
    ann_path = coco_root / "annotations" / "instances_val2017.json"
    img_dir = coco_root / "val2017"
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO 标注缺失: {ann_path}。请先运行: python main.py prepare-data --coco")
    log.info(f"[dataset] 解析 COCO 标注 {ann_path.name} ...")
    coco = _parse_coco_json(str(ann_path))
    target_ids = set(bundle.target_class_ids)
    image_size = bundle.image_size

    sizes = {
        "attack_train": int(cfg.dataset.attack_train_size),
        "attack_val": int(cfg.dataset.attack_val_size),
        "defense_train": int(cfg.dataset.defense_train_size),
        "defense_val": int(cfg.dataset.defense_val_size),
        "hidden_test": int(cfg.dataset.hidden_test_size),
    }
    buckets = {"attack_train": [], "attack_val": [], "defense_train": [], "defense_val": [], "hidden_test": []}
    # 按 image_id 哈希分桶, 保证划分稳定且互斥
    for im_id, im in coco["images"].items():
        anns = coco["ann_by_image"].get(im_id, [])
        if not any(a["category_id"] in target_ids for a in anns):
            continue
        b = _split_key(im_id, 100)
        if b < 20:
            bk = "attack_train"
        elif b < 30:
            bk = "attack_val"
        elif b < 60:
            bk = "defense_train"
        elif b < 75:
            bk = "defense_val"
        else:
            bk = "hidden_test"
        buckets[bk].append((im_id, im, anns))

    def fill(dst_list, items, limit):
        for im_id, im, anns in items[:limit]:
            fp = img_dir / im["file_name"]
            if not fp.exists():
                continue
            img = load_image(str(fp))
            img_lb, s, pad_x, pad_y = letterbox(img, image_size)
            dst_list.append(_make_sample(img_lb, im_id, anns, s, pad_x, pad_y, target_ids, str(fp)))

    fill(bundle.attack_train, buckets["attack_train"], sizes["attack_train"])
    fill(bundle.attack_val, buckets["attack_val"], sizes["attack_val"])
    fill(bundle.defense_train_clean, buckets["defense_train"], sizes["defense_train"])
    fill(bundle.defense_val_clean, buckets["defense_val"], sizes["defense_val"])
    fill(bundle.hidden_test_clean, buckets["hidden_test"], sizes["hidden_test"])
    bundle.source = "coco"


def _load_synthetic(bundle: DatasetBundle, syn_root: Path, cfg: Config):
    """读取合成数据集(COCO 格式目录)。若不存在则提示生成。"""
    splits = {
        "attack_train": (bundle.attack_train, "train", int(cfg.dataset.attack_train_size)),
        "attack_val": (bundle.attack_val, "val", int(cfg.dataset.attack_val_size)),
        "defense_train": (bundle.defense_train_clean, "train", int(cfg.dataset.defense_train_size)),
        "defense_val": (bundle.defense_val_clean, "val", int(cfg.dataset.defense_val_size)),
        "hidden_test": (bundle.hidden_test_clean, "hidden", int(cfg.dataset.hidden_test_size)),
    }
    # 合成数据按 split 目录组织: images/<split>/, annotations/instances_<split>.json
    need_generate = False
    for _, (_, split, _) in splits.items():
        if not (syn_root / "annotations" / f"instances_{split}.json").exists():
            need_generate = True
            break
    if need_generate:
        log.info("[dataset] 合成数据集缺失, 自动生成中...")
        from data.make_synthetic_dataset import generate_synthetic_dataset
        generate_synthetic_dataset(syn_root, cfg)

    target_ids = set(bundle.target_class_ids)
    image_size = bundle.image_size
    for _, (dst, split, limit) in splits.items():
        ann_path = syn_root / "annotations" / f"instances_{split}.json"
        img_dir = syn_root / "images" / split
        coco = _parse_coco_json(str(ann_path))
        count = 0
        for im_id, im in coco["images"].items():
            if count >= limit:
                break
            anns = coco["ann_by_image"].get(im_id, [])
            fp = img_dir / im["file_name"]
            if not fp.exists():
                continue
            img = load_image(str(fp))
            # 合成图本就是 image_size 方形, letterbox 仍调用以保持一致
            img_lb, s, pad_x, pad_y = letterbox(img, image_size)
            dst.append(_make_sample(img_lb, im_id, anns, s, pad_x, pad_y, target_ids, str(fp)))
            count += 1
    bundle.source = "synthetic"
