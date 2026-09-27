"""攻防方法注册表: 自动发现内置方法 + 加载外部选手提交。

支持形式:
- 攻击: BaseAttack 子类 (有 .name) 或裸函数 attack(env, task)->dict
- 防御: BaseDefense 子类 / DefenseModel 类 / 裸函数 defend(env, images)->ndarray
"""
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from pathlib import Path
from typing import Dict, List

from attacks.base import BaseAttack, as_attack
from defenses.base import BaseDefense, as_defense

ATTACKS: Dict[str, type] = {}
DEFENSES: Dict[str, type] = {}


def register_attack(name: str, obj):
    ATTACKS[name] = as_attack(obj)


def register_defense(name: str, obj):
    DEFENSES[name] = as_defense(obj)


def discover_builtin():
    """扫描 attacks/ 与 defenses/ 包, 注册所有 Base 子类。"""
    import attacks as _atk_pkg
    import defenses as _def_pkg
    for finder, name, ispkg in pkgutil.iter_modules(_atk_pkg.__path__):
        if name == "base":
            continue
        mod = importlib.import_module(f"attacks.{name}")
        for attr in dir(mod):
            v = getattr(mod, attr)
            if isinstance(v, type) and issubclass(v, BaseAttack) and v is not BaseAttack:
                register_attack(v.name, v)
    for finder, name, ispkg in pkgutil.iter_modules(_def_pkg.__path__):
        if name == "base":
            continue
        mod = importlib.import_module(f"defenses.{name}")
        for attr in dir(mod):
            v = getattr(mod, attr)
            if isinstance(v, type) and issubclass(v, BaseDefense) and v is not BaseDefense:
                register_defense(v.name, v)


# ----------------- 构造 (从 config 注入超参) -----------------

def build_attack(name: str, cfg):
    if name not in ATTACKS:
        raise KeyError(f"未知攻击方法: {name}; 可用: {list(ATTACKS)}")
    cls = ATTACKS[name]
    kwargs = {"seed": int(getattr(getattr(cfg, "run", None), "seed", 42))}
    acfg = getattr(getattr(cfg, "training", None), "attack", None)
    if name == "gradient_patch" and acfg:
        kwargs.update(dict(
            iters=int(getattr(acfg, "iters", 20)),
            lr=float(getattr(acfg, "lr", 0.03)),
            batch_size=int(getattr(acfg, "batch_size", 4)),
            tv_weight=float(getattr(acfg, "tv_weight", 0.01)),
        ))
    return cls(**kwargs)


def build_defense(name: str, cfg):
    if name not in DEFENSES:
        raise KeyError(f"未知防御方法: {name}; 可用: {list(DEFENSES)}")
    return DEFENSES[name]()


def list_attacks() -> List[str]:
    return sorted(ATTACKS.keys())


def list_defenses() -> List[str]:
    return sorted(DEFENSES.keys())


# ----------------- 外部提交加载 -----------------

def _load_module_from_file(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_attack_submission(dir_path: str):
    """加载选手攻击提交目录 (含 attack.py)。返回 BaseAttack 实例。"""
    p = Path(dir_path) / "attack.py"
    if not p.exists():
        raise FileNotFoundError(f"攻击提交缺少 attack.py: {p}")
    mod = _load_module_from_file("user_attack", str(p))
    if hasattr(mod, "attack"):
        return as_attack(mod.attack)
    for attr in dir(mod):
        v = getattr(mod, attr)
        if isinstance(v, type) and issubclass(v, BaseAttack) and v is not BaseAttack:
            return v()
    raise ValueError(f"{p} 未找到 attack 函数或 BaseAttack 子类")


def load_defense_submission(dir_path: str):
    """加载选手防御提交目录 (含 defense.py)。

    识别形式 (赛题 §29):
      - 裸函数 defend(env, images)->ndarray
      - BaseDefense 子类 (有 fit)
      - DefenseModel 类 (有 predict; 子类或鸭子类型) -> 包装, fit 时实例化(预训练, 不再训练)
    返回 BaseDefense 类。
    """
    from defenses.base import BaseDefense, as_defense
    p = Path(dir_path) / "defense.py"
    if not p.exists():
        raise FileNotFoundError(f"防御提交缺少 defense.py: {p}")
    mod = _load_module_from_file("user_defense", str(p))
    if hasattr(mod, "defend"):
        return as_defense(mod.defend)
    # BaseDefense 子类 (有 fit)
    for attr in dir(mod):
        v = getattr(mod, attr)
        if isinstance(v, type) and issubclass(v, BaseDefense) and v is not BaseDefense:
            return v
    # DefenseModel 类 (子类或鸭子类型: 有 predict) -> 包装为 BaseDefense, fit 实例化
    model_path = Path(dir_path) / "model" / "weights.pt"
    model_path = str(model_path) if model_path.exists() else None
    for attr in dir(mod):
        if attr.startswith("_"):
            continue
        v = getattr(mod, attr)
        if isinstance(v, type) and hasattr(v, "predict") and callable(getattr(v, "predict")):
            class _SubWrap(BaseDefense):
                name = v.__name__

                def fit(self, denv, clean_images, patched_images, cfg):
                    try:
                        return v(model_path) if model_path else v()
                    except TypeError:
                        return v()
            return _SubWrap
    raise ValueError(f"{p} 未找到 defend 函数或 DefenseModel/BaseDefense 子类")
