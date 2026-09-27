"""配置加载: 支持 yaml / json, 属性访问 (DotDict)。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """支持属性访问的字典, 嵌套在构造/赋值时递归转为 DotDict (原地存储),
    从而 cfg.methods.defenses = [...] 这类嵌套赋值能持久化。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = DotDict._wrap(v)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name: str, value: Any) -> None:
        # 类上定义了 data descriptor (如 property) 时优先走它, 否则存入 dict
        desc = getattr(type(self), name, None)
        if desc is not None and hasattr(desc, "__set__"):
            desc.__set__(self, value)
        else:
            self[name] = DotDict._wrap(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

    @staticmethod
    def _wrap(v: Any) -> Any:
        if isinstance(v, DotDict):
            return v
        if isinstance(v, dict):
            return DotDict(v)  # __init__ 递归转换
        if isinstance(v, list):
            return [DotDict._wrap(x) for x in v]
        return v

    def to_dict(self) -> dict:
        """递归转回普通 dict(用于序列化)。"""
        out = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, DotDict) else x for x in v]
            else:
                out[k] = v
        return out

    def merge(self, other: dict) -> "DotDict":
        """浅合并(顶层键覆盖)。"""
        for k, v in (other or {}).items():
            self[k] = DotDict._wrap(v)
        return self


class Config(DotDict):
    """顶层配置对象。"""

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Config":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text) or {}
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            # 兜底: 尝试 yaml(兼容 json)
            data = yaml.safe_load(text) or {}
        return cls(DotDict._wrap(data))

    @property
    def project_root(self) -> Path:
        """配置文件所在目录视为项目根。"""
        return Path(getattr(self, "_project_root", ".")).resolve()

    @project_root.setter
    def project_root(self, p: str | os.PathLike) -> None:
        self["_project_root"] = str(p)


def load_config(path: str | os.PathLike = "config.yaml") -> Config:
    """加载配置文件并锚定项目根目录。"""
    cfg = Config.from_file(path)
    cfg.project_root = Path(path).resolve().parent
    return cfg


def resolve_path(cfg: Config, p: str | os.PathLike) -> Path:
    """相对路径相对项目根解析; 绝对路径原样返回。"""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (cfg.project_root / pp).resolve()
