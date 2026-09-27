"""评测指标 (严格对齐赛题 §10/§20-§42)。

- IoU / 匹配
- Recall_clean/adv/occ, HideRate
- AttackGain, ConfidenceDrop, AttackScore_base
- TPR/FPR/TNR/BalancedAccuracy/AUROC
- EvasionRate, AttackScore_{i,j}, FinalAttackScore, FinalDefenseScore
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-8


# ----------------- IoU / 匹配 -----------------

def iou_xyxy(box_a, box_b) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a = (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1]))
    b = (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1]))
    return inter / (a + b - inter + EPS)


def best_match(gt_bbox, gt_cid, preds: Dict, iou_thresh: float, score_thresh: float,
               class_aware: bool = True) -> Tuple[bool, float]:
    """在 preds={boxes,labels,scores} 中找与 gt 最佳匹配。
    返回 (是否命中, 命中预测的置信度; 未命中则最佳匹配分数或 0)。
    命中条件: 类别一致(若 class_aware) & IoU>=iou_thresh & score>=score_thresh。
    """
    boxes = preds.get("boxes")
    if boxes is None or len(boxes) == 0:
        return False, 0.0
    labels = preds.get("labels")
    scores = preds.get("scores")
    best_iou = 0.0
    best_score = 0.0
    hit = False
    for i in range(len(boxes)):
        if class_aware and int(labels[i]) != int(gt_cid):
            continue
        if float(scores[i]) < score_thresh:
            continue
        v = iou_xyxy(gt_bbox, boxes[i])
        if v > best_iou:
            best_iou = v
            best_score = float(scores[i])
            if v >= iou_thresh:
                hit = True
    return hit, best_score


# ----------------- 攻击指标 -----------------

def evaluate_targets(eligible_targets: List[dict], preds_by_image: Dict,
                     iou_thresh: float, score_thresh: float, class_aware: bool = True):
    """对 eligible 目标统计命中数与每目标最佳匹配置信度。
    eligible_targets: list[{image_id, bbox, cid}]
    preds_by_image: {image_id: {boxes, labels, scores}}
    返回 dict{n_eligible, n_correct, recall, confs(list[float])}
    """
    n_eligible = len(eligible_targets)
    n_correct = 0
    confs = []
    for t in eligible_targets:
        preds = preds_by_image.get(t["image_id"], {"boxes": np.zeros((0, 4)),
                                                    "labels": np.zeros((0,)),
                                                    "scores": np.zeros((0,))})
        hit, score = best_match(t["bbox"], t["cid"], preds, iou_thresh, score_thresh, class_aware)
        if hit:
            n_correct += 1
        confs.append(score)
    recall = n_correct / max(1, n_eligible)
    return {"n_eligible": n_eligible, "n_correct": n_correct, "recall": recall, "confs": confs}


def hide_rate(recall: float) -> float:
    return 1.0 - recall


def attack_gain(hide_adv: float, hide_occ: float) -> float:
    """§23: max(0, (Hide_adv - Hide_occ) / (1 - Hide_occ + eps)), 裁剪 [0,1]。"""
    if hide_occ >= 1.0 - EPS:
        return 0.0 if hide_adv <= hide_occ else min(1.0, (hide_adv - hide_occ) / EPS)
    g = max(0.0, (hide_adv - hide_occ) / (1.0 - hide_occ + EPS))
    return float(min(1.0, max(0.0, g)))


def confidence_drop(conf_clean_list: List[float], conf_adv_list: List[float]) -> float:
    """§24: 对每个 eligible 目标, max(0, (conf_clean - conf_adv)/(conf_clean+eps)), 取均值。"""
    drops = []
    for cc, ca in zip(conf_clean_list, conf_adv_list):
        d = max(0.0, (cc - ca) / (cc + EPS))
        drops.append(d)
    return float(np.mean(drops)) if drops else 0.0


def attack_score_base(gain: float, conf_drop: float) -> float:
    """§25: 0.85*Gain + 0.15*ConfDrop。"""
    return 0.85 * gain + 0.15 * conf_drop


# ----------------- 防御指标 -----------------

def tpr_fpr_tnr(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Dict:
    """y_true: 0/1, y_pred_bin: 0/1 (阈值化后)。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred_bin = np.asarray(y_pred_bin).astype(int)
    tp = int(np.sum((y_pred_bin == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_bin == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_bin == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred_bin == 0) & (y_true == 0)))
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    tnr = tn / max(1, fp + tn)
    bal = (tpr + tnr) / 2.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "tpr": float(tpr),
            "fpr": float(fpr), "tnr": float(tnr), "balanced_accuracy": float(bal)}


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """§35。仅一个类别时返回 0.5。"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def defense_score(balanced_accuracy: float, auroc_val: float) -> float:
    """§36: 0.7*BalAcc + 0.3*AUROC。"""
    return 0.7 * balanced_accuracy + 0.3 * auroc_val


def evasion_rate(tpr: float) -> float:
    """§38: 1 - TPR。"""
    return 1.0 - tpr


def attack_score_matchup(attack_base: float, evasion: float) -> float:
    """§39: base * (0.8 + 0.2*Evasion)。"""
    return attack_base * (0.8 + 0.2 * evasion)


def final_attack_score(matchup_scores: List[float]) -> float:
    """§41: 对各防御取平均。"""
    return float(np.mean(matchup_scores)) if matchup_scores else 0.0


def final_defense_score(matchup_defense_scores: List[float]) -> float:
    """§42: 对各攻击取平均。"""
    return float(np.mean(matchup_defense_scores)) if matchup_defense_scores else 0.0
