"""通用工具: 随机种子、日志、图像 IO、tensor/numpy 转换。"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_LOG_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    global _LOG_CONFIGURED
    logger = logging.getLogger("patch_ctf")
    if not _LOG_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _LOG_CONFIGURED = True
    return logger


log = setup_logging()


def set_seed(seed: int) -> None:
    """统一设置 python / numpy / torch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # 本环境为 CPU, 兼容分支
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------- 图像 IO 与转换 -----------------

def load_image(path: str | os.PathLike) -> np.ndarray:
    """读 RGB 图像为 HxWx3 uint8 ndarray。"""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(arr: np.ndarray, path: str | os.PathLike) -> None:
    """保存 HxWx3 uint8 / float[0,1] ndarray 为图像。"""
    from PIL import Image
    arr = to_uint8(arr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """float[0,1] 或 uint8 -> uint8。"""
    if arr.dtype == np.uint8:
        return arr
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def to_float01(arr: np.ndarray) -> np.ndarray:
    """uint8/任意 -> float32 [0,1], HWC。"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.dtype == np.uint8 or arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def hwc_to_chw(arr: np.ndarray) -> np.ndarray:
    """HxWxC -> CxHxW。"""
    return np.transpose(arr, (2, 0, 1))


def chw_to_hwc(arr: np.ndarray) -> np.ndarray:
    """CxHxW -> HxWxC。"""
    if arr.ndim == 2:
        return arr
    return np.transpose(arr, (1, 2, 0))


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(obj: Any, path: str | os.PathLike, indent: int = 2) -> None:
    import json
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_default, ensure_ascii=False)
