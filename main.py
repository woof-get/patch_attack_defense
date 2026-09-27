"""目标检测对抗贴片攻防对抗赛 - 后端 CLI。

命令:
  python main.py prepare-data [--coco|--synthetic] [--max-images N]   下载/生成数据集
  python main.py list                                                 列出已注册攻防方法
  python main.py run [--config ...] [--attacks a,b] [--defenses d,b] [--seed S]
                                                                      全对阵评测
  python main.py matchup -a <attack> -d <defense>                     单对阵
  python main.py submission -a <attack_dir> -d <defense_dir>          加载选手提交跑对阵
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 确保项目根在 sys.path (从任意目录运行均可)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from judge.config import load_config
from judge.utils import log, set_seed
from judge import registry


def cmd_prepare_data(args):
    cfg = load_config(args.config)
    if args.synthetic:
        from data.make_synthetic_dataset import generate_synthetic_dataset
        syn_root = (cfg.project_root / cfg.dataset.synthetic_root).resolve()
        generate_synthetic_dataset(syn_root, cfg, seed=int(getattr(cfg.run, "seed", 42)))
    elif args.coco:
        from data.download_coco import download_val2017_subset, download_val2017_full
        from judge.config import resolve_path
        out = resolve_path(cfg, cfg.dataset.root)
        if args.full:
            download_val2017_full(out)
        else:
            classes = list(cfg.dataset.target_classes)
            download_val2017_subset(out, classes, args.max_images)
        log.info(f"[prepare-data] COCO 已下载到 {out}")
    else:
        # 默认: 尝试 COCO 子集, 失败回退合成
        from judge.config import resolve_path
        out = resolve_path(cfg, cfg.dataset.root)
        try:
            from data.download_coco import download_val2017_subset
            classes = list(cfg.dataset.target_classes)
            download_val2017_subset(out, classes, args.max_images)
            log.info("[prepare-data] COCO 子集下载完成")
        except Exception as e:
            log.warning(f"[prepare-data] COCO 下载失败 ({e}), 回退合成数据集")
            from data.make_synthetic_dataset import generate_synthetic_dataset
            syn_root = (cfg.project_root / cfg.dataset.synthetic_root).resolve()
            generate_synthetic_dataset(syn_root, cfg, seed=int(getattr(cfg.run, "seed", 42)))


def cmd_list(args):
    registry.discover_builtin()
    print("攻击方法:", registry.list_attacks())
    print("防御方法:", registry.list_defenses())


def cmd_run(args):
    from judge.evaluator import Evaluator
    cfg = load_config(args.config)
    if args.attacks:
        cfg.methods.attacks = args.attacks.split(",")
    if args.defenses:
        cfg.methods.defenses = args.defenses.split(",")
    if args.seed is not None:
        cfg.run.seed = args.seed
    ev = Evaluator(cfg)
    ev.registry = registry
    registry.discover_builtin()
    res = ev.run_all_vs_all()
    ev.save(getattr(cfg.run, "output_dir", "results"))
    _print_summary(res)


def cmd_matchup(args):
    registry.discover_builtin()
    from judge.evaluator import Evaluator
    cfg = load_config(args.config)
    ev = Evaluator(cfg)
    ev.registry = registry
    res = ev.run_all_vs_all(attack_names=[args.attack], defense_names=[args.defense])
    ev.save(getattr(cfg.run, "output_dir", "results"))
    _print_summary(res)


def cmd_submission(args):
    registry.discover_builtin()
    atk = registry.load_attack_submission(args.attack_dir)
    defense = registry.load_defense_submission(args.defense_dir)
    from judge.evaluator import Evaluator
    cfg = load_config(args.config)
    ev = Evaluator(cfg)
    ev.registry = registry
    # 注册为临时方法
    registry.ATTACKS["__submission__"] = type(atk) if not isinstance(atk, type) else atk
    registry.DEFENSES["__submission__"] = type(defense) if not isinstance(defense, type) else defense
    res = ev.run_all_vs_all(attack_names=["__submission__"], defense_names=["__submission__"])
    ev.save(getattr(cfg.run, "output_dir", "results"))
    _print_summary(res)


def _print_summary(res):
    print("\n" + "=" * 60)
    print(f"数据集: {res['dataset']}  检测器: {res['detectors']}")
    print(f"合格: 攻击 {res['qualified_counts']['attacks']} / 防御 {res['qualified_counts']['defenses']}")
    print("\n--- 攻击排名 ---")
    for i, r in enumerate(res["rankings"]["attacks"]):
        print(f"  {i+1}. {r['name']:24s} FinalAttackScore={r['final_score']:.4f} "
              f"(AttackGain={r.get('attack_gain',0):.3f}) {'[qualified]' if r.get('qualified') else ''}")
    print("\n--- 防御排名 ---")
    for i, r in enumerate(res["rankings"]["defenses"]):
        print(f"  {i+1}. {r['name']:24s} FinalDefenseScore={r['final_score']:.4f} "
              f"(Val={r.get('defense_score_val',0):.3f}) {'[qualified]' if r.get('qualified') else ''}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="目标检测对抗贴片攻防对抗后端")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data", help="下载/生成数据集")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--coco", action="store_true", help="下载 COCO")
    p.add_argument("--synthetic", action="store_true", help="生成合成数据集")
    p.add_argument("--full", action="store_true", help="下载完整 val2017")
    p.add_argument("--max-images", type=int, default=100)
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("list", help="列出攻防方法")
    p.add_argument("--config", default="config.yaml")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="全对阵评测")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--attacks", default=None)
    p.add_argument("--defenses", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("matchup", help="单对阵")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("-a", "--attack", required=True)
    p.add_argument("-d", "--defense", required=True)
    p.set_defaults(func=cmd_matchup)

    p = sub.add_parser("submission", help="加载选手提交跑对阵")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("-a", "--attack-dir", required=True)
    p.add_argument("-d", "--defense-dir", required=True)
    p.set_defaults(func=cmd_submission)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
