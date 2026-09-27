"""目标检测对抗贴片攻防对抗赛 - 裁判程序后端。"""
from .config import Config, load_config
from .env import AttackEnv, DefenseEnv

__all__ = ["Config", "load_config", "AttackEnv", "DefenseEnv"]
__version__ = "1.0.0"
