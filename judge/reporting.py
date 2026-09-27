"""结果输出: summary.json / 矩阵 CSV / 攻防矩阵热力图 (HTML+PNG) / 排名。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .utils import ensure_dir, json_dump, log


def _matrix_to_rows(matrix, row_names, col_names, key):
    rows = []
    for rn in row_names:
        row = {"attack": rn}
        for cn in col_names:
            v = matrix.get(rn, {}).get(cn, {})
            row[cn] = v.get(key, "") if isinstance(v, dict) else ""
        rows.append(row)
    return rows


def _write_csv(rows, path, col_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack"] + col_names)
        for r in rows:
            w.writerow([r["attack"]] + [r[c] for c in col_names])


def _matrix_array(matrix, atk_names, def_names, key):
    arr = np.full((len(atk_names), len(def_names)), np.nan)
    for i, a in enumerate(atk_names):
        for j, d in enumerate(def_names):
            v = matrix.get(a, {}).get(d, {})
            if isinstance(v, dict) and key in v:
                arr[i, j] = v[key]
    return arr


def _heatmap(arr, atk_names, def_names, title, path_png, path_html, cmap="RdYlGn"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(def_names) + 2),
                                    max(3, 0.8 * len(atk_names) + 2)))
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(def_names))); ax.set_xticklabels(def_names, rotation=30, ha="right")
    ax.set_yticks(range(len(atk_names))); ax.set_yticklabels(atk_names)
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                txt = "-"
            else:
                txt = f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path_png, dpi=120)
    plt.close(fig)
    # HTML 包装
    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title></head>
<body><h2>{title}</h2><img src='{Path(path_png).name}' /></body></html>"""
    Path(path_html).write_text(html, encoding="utf-8")


def save_results(results: dict, out_dir: str):
    out = Path(out_dir)
    ensure_dir(out)
    # 1) summary.json
    json_dump(results, out / "summary.json")
    # 2) 排名
    atk_rank = results.get("rankings", {}).get("attacks", [])
    def_rank = results.get("rankings", {}).get("defenses", [])
    with open(out / "rankings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "attack", "final_attack_score", "attack_gain", "qualified"])
        for i, r in enumerate(atk_rank):
            w.writerow([i + 1, r["name"], f"{r['final_score']:.4f}",
                        f"{r.get('attack_gain', 0):.4f}", r.get("qualified", False)])
        w.writerow([])
        w.writerow(["rank", "defense", "final_defense_score", "defense_score_val", "qualified"])
        for i, r in enumerate(def_rank):
            w.writerow([i + 1, r["name"], f"{r['final_score']:.4f}",
                        f"{r.get('defense_score_val', 0):.4f}", r.get("qualified", False)])
    # 3) 矩阵 CSV
    matrix = results.get("matrix", {})
    atk_names = list(matrix.keys())
    def_names = sorted({d for a in matrix.values() for d in a.keys()}) if matrix else []
    if atk_names and def_names:
        for key, fname in [("attack_score", "matrix_attack_score.csv"),
                           ("defense_score", "matrix_defense_score.csv"),
                           ("evasion", "matrix_evasion.csv"),
                           ("tpr", "matrix_tpr.csv")]:
            rows = _matrix_to_rows(matrix, atk_names, def_names, key)
            _write_csv(rows, out / fname, def_names)
        # 4) 热力图
        try:
            arr = _matrix_array(matrix, atk_names, def_names, "attack_score")
            _heatmap(arr, atk_names, def_names, "Attack Score (Attack × Defense)",
                     str(out / "heatmap_attack.png"), str(out / "heatmap.html"))
            arr2 = _matrix_array(matrix, atk_names, def_names, "tpr")
            _heatmap(arr2, atk_names, def_names, "Defense TPR (Attack × Defense)",
                     str(out / "heatmap_defense_tpr.png"), str(out / "heatmap_tpr.html"),
                     cmap="YlGnBu")
        except Exception as e:
            log.warning(f"[reporting] 热力图生成失败: {e}")
    # 5) 单方法明细 CSV
    with open(out / "attack_details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack", "valid", "attack_gain", "hide_adv", "hide_occ",
                    "conf_drop", "attack_score_base", "qualified"])
        for a in results.get("attacks", {}).values():
            w.writerow([a.get("name"), a.get("valid"), f"{a.get('attack_gain', 0):.4f}",
                        f"{a.get('hide_adv', 0):.4f}", f"{a.get('hide_occ', 0):.4f}",
                        f"{a.get('conf_drop', 0):.4f}", f"{a.get('attack_score_base', 0):.4f}",
                        a.get("qualified")])
    with open(out / "defense_details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["defense", "trained", "tpr", "fpr", "balanced_accuracy",
                    "auroc", "defense_score", "qualified"])
        for d in results.get("defenses", {}).values():
            w.writerow([d.get("name"), d.get("trained"), f"{d.get('tpr', 0):.4f}",
                        f"{d.get('fpr', 0):.4f}", f"{d.get('balanced_accuracy', 0):.4f}",
                        f"{d.get('auroc', 0):.4f}", f"{d.get('defense_score', 0):.4f}",
                        d.get("qualified")])
    log.info(f"[reporting] 已输出 summary.json / rankings.csv / matrix_*.csv / heatmap.png")
