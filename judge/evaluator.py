"""评测器: 攻防对抗全流水线编排 (赛题 §50/§52/§65)。

流程:
  1. 加载数据集 + 构建检测器
  2. 各检测器上计算 eligible 目标 (干净可检测)
  3. 每个攻击: 生成 patch -> 合法性检查 -> 渲染 adv/occ -> 检测评测 -> AttackScore_base
  4. 用基线攻击池渲染防御训练数据 (clean + A0-patched)
  5. 每个防御: 训练 -> defense_val 上 TPR/FPR/BalAcc/AUROC -> DefenseScore
  6. 全对阵 A×D: 每对 Evasion/AttackScore_{i,j}/DefenseScore
  7. 聚合最终得分 + 资格线 -> 排名

攻击与防御解耦: 攻击只产 patch, 防御只产概率, 裁判统一渲染与评测。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import Config, resolve_path
from .utils import log, set_seed, json_dump
from .dataset import load_dataset, DatasetBundle
from .detector import build_detectors, Detector
from .tiny_detector import TinyDetector
from .patch_renderer import apply_patch, apply_occlusion, sample_params, render_patch
from .attack_validator import validate_patch
from .env import build_attack_env, build_defense_env, AttackEnv, DefenseEnv
from .defense_runner import DefenseRunner
from . import metrics as M
from . import registry


class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bundle: Optional[DatasetBundle] = None
        self.detectors: List[Detector] = []
        self.eligible_by_det: Dict[str, list] = {}
        self.results: dict = {}

    @classmethod
    def from_config(cls, cfg_path: str = "config.yaml"):
        from .config import load_config
        cfg = load_config(cfg_path)
        ev = cls(cfg)
        ev.registry = registry
        registry.discover_builtin()
        return ev

    # ----------------- 检测器构建 -----------------
    def _build_detectors(self):
        bundle = self.bundle
        if bundle.source == "synthetic":
            syn_root = resolve_path(self.cfg, self.cfg.dataset.synthetic_root)
            weights = syn_root / "detector.pt"
            det = TinyDetector(
                target_class_ids=bundle.target_class_ids,
                device=getattr(self.cfg.detectors, "device", "cpu"),
                score_thresh=float(self.cfg.detectors.model_a.score_thresh),
                nms_iou=float(self.cfg.detectors.model_a.nms_iou),
                input_size=bundle.image_size,
                weights_path=str(weights) if weights.exists() else None,
            )
            det.name = "tiny_detector(Model-A)"
            self.detectors = [det]
        else:
            self.detectors = build_detectors(self.cfg)
        log.info(f"[eval] 检测器: {[d.name for d in self.detectors]}")

    # ----------------- eligible 目标 -----------------
    def _compute_eligible(self, detector: Detector, samples: list) -> list:
        """干净图过检测器, 保留 IoU>=0.5 & score>=τ & 类别正确的目标 (赛题 §10)。"""
        iou_th = float(self.cfg.eval.det_iou_thresh)
        score_th = float(self.cfg.eval.det_score_thresh)
        target_ids = set(self.bundle.target_class_ids)
        eligible = []
        max_tgt = int(self.cfg.run.max_eligible_targets)
        for s in samples:
            img = s["image"]
            preds = detector.predict(img)[0]
            for ann in s["annotations"]:
                cid = int(ann["category_id"])
                if cid not in target_ids:
                    continue
                hit, _ = M.best_match(ann["bbox"], cid, preds, iou_th, score_th, class_aware=True)
                if hit:
                    eligible.append({"image_id": s["image_id"], "bbox": ann["bbox"], "cid": cid,
                                     "image": img, "category_name": ann.get("category_name", "")})
                    if len(eligible) >= max_tgt:
                        return eligible
        return eligible

    # ----------------- 单攻击评测 -----------------
    def _eval_attack_on_detector(self, detector: Detector, eligible: list, patch: np.ndarray,
                                 rng) -> dict:
        """对 eligible 目标渲染 adv/occ, 推理, 算 Recall/HideRate/AttackGain/ConfDrop。"""
        iou_th = float(self.cfg.eval.det_iou_thresh)
        score_th = float(self.cfg.eval.det_score_thresh)
        # 干净预测 (eligible 已是干净可检测, recall_clean≈1; 仍记录 conf)
        preds_clean_by_img = {}
        imgs_by_id = {}
        for t in eligible:
            if t["image_id"] not in preds_clean_by_img:
                preds_clean_by_img[t["image_id"]] = detector.predict(t["image"])[0]
                imgs_by_id[t["image_id"]] = t["image"]

        # 渲染 adv / occ (同一目标同一参数)
        adv_imgs_by_id, occ_imgs_by_id = {}, {}
        adv_params = []
        for t in eligible:
            iid = t["image_id"]
            if iid not in adv_imgs_by_id:
                a_img, params = apply_patch(imgs_by_id[iid], t["bbox"], patch, rng, self.cfg)
                o_img, _ = apply_occlusion(imgs_by_id[iid], t["bbox"], params, rng, self.cfg)
                adv_imgs_by_id[iid] = a_img
                occ_imgs_by_id[iid] = o_img
                adv_params.append(params)

        # 推理
        preds_adv, preds_occ = {}, {}
        for iid in imgs_by_id:
            preds_adv[iid] = detector.predict(adv_imgs_by_id[iid])[0]
            preds_occ[iid] = detector.predict(occ_imgs_by_id[iid])[0]

        clean_eval = M.evaluate_targets(eligible, preds_clean_by_img, iou_th, score_th, class_aware=True)
        adv_eval = M.evaluate_targets(eligible, preds_adv, iou_th, score_th, class_aware=True)
        occ_eval = M.evaluate_targets(eligible, preds_occ, iou_th, score_th, class_aware=True)

        hide_adv = M.hide_rate(adv_eval["recall"])
        hide_occ = M.hide_rate(occ_eval["recall"])
        gain = M.attack_gain(hide_adv, hide_occ)
        conf_drop = M.confidence_drop(clean_eval["confs"], adv_eval["confs"])
        return {
            "n_eligible": len(eligible),
            "recall_clean": clean_eval["recall"],
            "recall_adv": adv_eval["recall"],
            "recall_occ": occ_eval["recall"],
            "hide_adv": hide_adv,
            "hide_occ": hide_occ,
            "attack_gain": gain,
            "conf_drop": conf_drop,
        }

    def evaluate_attack(self, name: str, attack) -> dict:
        """生成 patch + 合法性检查 + 多检测器评测 -> AttackScore_base。"""
        patch_size = int(self.cfg.attack.patch_size)
        # 构建攻击 task
        images = [s["image"] for s in self.bundle.attack_train]
        annotations = [[{"bbox": a["bbox"], "category_id": int(a["category_id"])}
                        for a in s["annotations"]] for s in self.bundle.attack_train]
        target_class = int(self.bundle.target_class_ids[0])
        a_env = build_attack_env(self.cfg, self.detectors[0], self.bundle)
        task = {
            "images": images,
            "annotations": annotations,
            "target_class": target_class,
            "patch_size": patch_size,
            "max_patch_area_ratio": float(self.cfg.attack.max_patch_area_ratio),
            "detector": a_env.detector,
        }
        log.info(f"[eval] === 攻击: {name} === 生成贴片...")
        try:
            result = attack(a_env, task)
        except Exception as e:
            import traceback
            log.error(f"[eval] 攻击 {name} 执行失败: {e}\n{traceback.format_exc()}")
            return {"name": name, "valid": False, "error": str(e)}
        ok, patch, msgs = validate_patch(result, patch_size)
        for m in msgs:
            log.warning(f"[eval] 攻击 {name}: {m}")
        if not ok:
            log.error(f"[eval] 攻击 {name} 贴片不合法: {msgs}")
            return {"name": name, "valid": False, "errors": msgs}

        rng = np.random.default_rng(self.cfg.run.seed + hash(name) % 100000)
        per_det = []
        for det in self.detectors:
            elig = self.eligible_by_det.get(det.name)
            if not elig:
                continue
            log.info(f"[eval] 攻击 {name} @ {det.name}: 渲染+推理 ({len(elig)} 目标)...")
            r = self._eval_attack_on_detector(det, elig, patch, rng)
            r["detector"] = det.name
            per_det.append(r)
            log.info(f"[eval]   {det.name}: Recall_clean={r['recall_clean']:.3f} "
                     f"Recall_adv={r['recall_adv']:.3f} Recall_occ={r['recall_occ']:.3f} "
                     f"HideRate_adv={r['hide_adv']:.3f} AttackGain={r['attack_gain']:.3f} "
                     f"ConfDrop={r['conf_drop']:.3f}")

        # 多检测器平均
        gain = float(np.mean([r["attack_gain"] for r in per_det])) if per_det else 0.0
        conf_drop = float(np.mean([r["conf_drop"] for r in per_det])) if per_det else 0.0
        hide_adv = float(np.mean([r["hide_adv"] for r in per_det])) if per_det else 0.0
        hide_occ = float(np.mean([r["hide_occ"] for r in per_det])) if per_det else 0.0
        base = M.attack_score_base(gain, conf_drop)
        qualified = (gain >= float(self.cfg.eval.attack_qual_gain)
                     and hide_adv >= float(self.cfg.eval.attack_qual_hide))
        return {
            "name": name, "valid": True, "patch_shape": list(patch.shape),
            "per_detector": per_det,
            "attack_gain": gain, "hide_adv": hide_adv, "hide_occ": hide_occ,
            "conf_drop": conf_drop, "attack_score_base": base,
            "qualified": qualified,
            "_patch": patch,  # 留作全对阵用
        }

    # ----------------- 防御训练数据 -----------------
    def _build_defense_training_data(self) -> tuple:
        """用基线攻击池 A0 渲染防御训练数据 (clean + patched)。返回 (clean_imgs, patched_imgs)。"""
        baseline_attacks = ["random_noise", "checkerboard", "gray_patch"]
        patch_size = int(self.cfg.attack.patch_size)
        rng = np.random.default_rng(self.cfg.run.seed)
        # 生成基线贴片 (这些攻击不依赖 env, 传 None)
        baseline_patches = []
        for bn in baseline_attacks:
            a = registry.build_attack(bn, self.cfg)
            ps = a(None, {"patch_size": patch_size})["patch"]
            baseline_patches.append(ps)
        # 多种随机初始化的噪声贴片 (增加多样性)
        for _ in range(5):
            baseline_patches.append(np.random.default_rng(rng.integers(1e9)).random(
                (3, patch_size, patch_size), dtype=np.float32))

        n = len(self.bundle.defense_train_clean)
        half = n // 2
        clean_imgs = [s["image"].copy() for s in self.bundle.defense_train_clean[:half]]
        patched_imgs = []
        for s in self.bundle.defense_train_clean[half:2 * half]:
            anns = s["annotations"]
            if not anns:
                continue
            ann = anns[int(rng.integers(len(anns)))]
            patch = baseline_patches[int(rng.integers(len(baseline_patches)))]
            p_img, _ = apply_patch(s["image"], ann["bbox"], patch, rng, self.cfg)
            patched_imgs.append(p_img)
        log.info(f"[eval] 防御训练数据: {len(clean_imgs)} clean + {len(patched_imgs)} patched")
        return clean_imgs, patched_imgs

    # ----------------- 防训练 + 验证 -----------------
    def evaluate_defense(self, name: str, defense, train_clean, train_patched) -> dict:
        denv = build_defense_env(self.cfg)
        log.info(f"[eval] === 防御: {name} === 训练...")
        try:
            model = defense.fit(denv, train_clean, train_patched, self.cfg)
        except Exception as e:
            import traceback
            log.error(f"[eval] 防御 {name} 训练失败: {e}\n{traceback.format_exc()}")
            return {"name": name, "trained": False, "error": str(e)}
        runner = DefenseRunner(model, env=denv)
        # 验证集: clean (FPR) + 基线 patched (TPR)
        rng = np.random.default_rng(self.cfg.run.seed + 7)
        val_clean = [s["image"].copy() for s in self.bundle.defense_val_clean]
        baseline_patches = []
        for bn in ["random_noise", "checkerboard", "gray_patch"]:
            a = registry.build_attack(bn, self.cfg)
            baseline_patches.append(a(None, {"patch_size": int(self.cfg.attack.patch_size)})["patch"])
        # 额外随机噪声贴片增加多样性
        for _ in range(5):
            baseline_patches.append(np.random.default_rng(rng.integers(1e9)).random(
                (3, int(self.cfg.attack.patch_size), int(self.cfg.attack.patch_size)), dtype=np.float32))
        val_patched = []
        for s in self.bundle.defense_val_clean:
            anns = s["annotations"]
            if not anns:
                continue
            ann = anns[int(rng.integers(len(anns)))]
            patch = baseline_patches[int(rng.integers(len(baseline_patches)))]
            p_img, _ = apply_patch(s["image"], ann["bbox"], patch, rng, self.cfg)
            val_patched.append(p_img)

        y_true = np.concatenate([np.zeros(len(val_clean)), np.ones(len(val_patched))])
        scores = np.concatenate([runner.predict_many(val_clean), runner.predict_many(val_patched)])
        y_bin = (scores >= 0.5).astype(int)
        tf = M.tpr_fpr_tnr(y_true, y_bin)
        au = M.auroc(y_true, scores)
        dscore = M.defense_score(tf["balanced_accuracy"], au)
        qualified = (tf["fpr"] <= float(self.cfg.eval.defense_qual_fpr)
                     and tf["tpr"] >= float(self.cfg.eval.defense_qual_tpr))
        log.info(f"[eval] 防御 {name}: TPR={tf['tpr']:.3f} FPR={tf['fpr']:.3f} "
                 f"BalAcc={tf['balanced_accuracy']:.3f} AUROC={au:.3f} "
                 f"DefenseScore={dscore:.3f} qualified={qualified}")
        return {
            "name": name, "trained": True,
            "tpr": tf["tpr"], "fpr": tf["fpr"], "tnr": tf["tnr"],
            "balanced_accuracy": tf["balanced_accuracy"], "auroc": au,
            "defense_score": dscore, "qualified": qualified,
            "_runner": runner,
        }

    # ----------------- 全对阵 -----------------
    def _evaluate_matchup(self, attack_res: dict, defense_res: dict, rng) -> dict:
        """攻击 patch × 防御模型: 渲染攻击图 -> 防御预测 -> TPR/Evasion/双方得分。"""
        patch = attack_res["_patch"]
        runner = defense_res["_runner"]
        # 用 hidden_test 干净图 + 攻击 patch 渲染 patched
        val_clean = [s["image"].copy() for s in self.bundle.hidden_test_clean]
        val_patched = []
        for s in self.bundle.hidden_test_clean:
            anns = s["annotations"]
            if not anns:
                val_patched.append(s["image"].copy())
                continue
            ann = anns[int(rng.integers(len(anns)))]
            p_img, _ = apply_patch(s["image"], ann["bbox"], patch, rng, self.cfg)
            val_patched.append(p_img)
        y_true = np.concatenate([np.zeros(len(val_clean)), np.ones(len(val_patched))])
        scores = np.concatenate([runner.predict_many(val_clean), runner.predict_many(val_patched)])
        y_bin = (scores >= 0.5).astype(int)
        tf = M.tpr_fpr_tnr(y_true, y_bin)
        au = M.auroc(y_true, scores)
        evasion = M.evasion_rate(tf["tpr"])
        atk_score = M.attack_score_matchup(attack_res["attack_score_base"], evasion)
        def_score = M.defense_score(tf["balanced_accuracy"], au)
        return {
            "tpr": tf["tpr"], "fpr": tf["fpr"], "tnr": tf["tnr"],
            "balanced_accuracy": tf["balanced_accuracy"], "auroc": au,
            "evasion": evasion, "attack_score": atk_score, "defense_score": def_score,
        }

    # ----------------- 主入口 -----------------
    def run_all_vs_all(self, attack_names=None, defense_names=None) -> dict:
        set_seed(int(self.cfg.run.seed))
        log.info("[eval] 加载数据集...")
        self.bundle = load_dataset(self.cfg)
        self._build_detectors()
        # eligible
        for det in self.detectors:
            log.info(f"[eval] 计算 eligible 目标 @ {det.name} (hidden_test)...")
            self.eligible_by_det[det.name] = self._compute_eligible(det, self.bundle.hidden_test_clean)
            log.info(f"[eval]   {det.name}: {len(self.eligible_by_det[det.name])} eligible 目标")

        attack_names = attack_names or list(self.cfg.methods.attacks)
        defense_names = defense_names or list(self.cfg.methods.defenses)

        # 1) 攻击评测
        attacks = {}
        for an in attack_names:
            attack = registry.build_attack(an, self.cfg)
            attacks[an] = self.evaluate_attack(an, attack)

        # 2) 防御训练数据 + 防御评测
        tr_clean, tr_patched = self._build_defense_training_data()
        defenses = {}
        for dn in defense_names:
            defense = registry.build_defense(dn, self.cfg)
            defenses[dn] = self.evaluate_defense(dn, defense, tr_clean, tr_patched)

        # 3) 全对阵矩阵
        rng = np.random.default_rng(self.cfg.run.seed + 1)
        matrix = {}
        for an, ar in attacks.items():
            if not ar.get("valid"):
                continue
            matrix[an] = {}
            for dn, dr in defenses.items():
                if not dr.get("trained"):
                    continue
                mu = self._evaluate_matchup(ar, dr, rng)
                matrix[an][dn] = mu
                log.info(f"[eval] 阵 {an} × {dn}: Evasion={mu['evasion']:.3f} "
                         f"AttackScore={mu['attack_score']:.3f} DefenseScore={mu['defense_score']:.3f}")

        # 4) 聚合排名 (优先合格方法; 无合格时用全部)
        q_attacks = [a for a in attacks.values() if a.get("valid") and a.get("qualified")]
        q_defenses = [d for d in defenses.values() if d.get("trained") and d.get("qualified")]
        atk_pool = q_attacks if q_attacks else [a for a in attacks.values() if a.get("valid")]
        def_pool = q_defenses if q_defenses else [d for d in defenses.values() if d.get("trained")]

        atk_ranking = []
        for ar in atk_pool:
            scores = [matrix.get(ar["name"], {}).get(dn, {}).get("attack_score", 0.0)
                      for dn in [d["name"] for d in def_pool]]
            final = M.final_attack_score(scores)
            atk_ranking.append({"name": ar["name"], "final_score": final,
                                "attack_gain": ar["attack_gain"], "qualified": ar["qualified"]})
        def_ranking = []
        for dr in def_pool:
            scores = [matrix.get(an, {}).get(dr["name"], {}).get("defense_score", 0.0)
                      for an in [a["name"] for a in atk_pool]]
            final = M.final_defense_score(scores)
            def_ranking.append({"name": dr["name"], "final_score": final,
                                "defense_score_val": dr["defense_score"], "qualified": dr["qualified"]})
        atk_ranking.sort(key=lambda x: x["final_score"], reverse=True)
        def_ranking.sort(key=lambda x: x["final_score"], reverse=True)

        # 清理内部对象
        for ar in attacks.values():
            ar.pop("_patch", None)
        for dr in defenses.values():
            dr.pop("_runner", None)

        self.results = {
            "dataset": self.bundle.source,
            "target_classes": self.bundle.target_class_names,
            "detectors": [d.name for d in self.detectors],
            "attacks": attacks,
            "defenses": defenses,
            "matrix": matrix,
            "rankings": {"attacks": atk_ranking, "defenses": def_ranking},
            "qualified_counts": {"attacks": len(q_attacks), "defenses": len(q_defenses)},
        }
        return self.results

    # ----------------- 保存 -----------------
    def save(self, out_dir: str = "results"):
        from .reporting import save_results
        out = resolve_path(self.cfg, out_dir)
        save_results(self.results, str(out))
        log.info(f"[eval] 结果已保存到 {out}")
        return str(out)
