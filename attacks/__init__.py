"""内置攻击方法池 A0。"""
from .base import BaseAttack, as_attack
from .random_noise import RandomNoiseAttack
from .checkerboard import CheckerboardAttack
from .gray_patch import GrayPatchAttack
from .gradient_patch import GradientPatchAttack

__all__ = ["BaseAttack", "as_attack", "RandomNoiseAttack", "CheckerboardAttack",
           "GrayPatchAttack", "GradientPatchAttack"]
