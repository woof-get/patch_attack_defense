# 目标检测对抗贴片攻防对抗赛 - 后端

基于赛题《目标检测对抗贴片生成与未知贴片检测攻防对抗赛》实现的攻防对抗后端。
对 **任意攻击方法 × 任意防御方法** 自动跑通：

```
攻击生成贴片 → 裁判统一渲染 → 目标检测评测 + 防御二分类评测 → 攻防矩阵 → 排名
```

**攻击与防御完全解耦**：攻击方只输出一张 universal patch；防御方只输出 图像→概率；
贴片位置/缩放/旋转/颜色变换全部由裁判端 `apply_patch` 统一负责（不提供给防御方）。

---

## 一、环境

- Python 3.11 / PyTorch 2.13 (CPU) / torchvision 0.28
- 已验证：检测器权重可从 PyTorch CDN 下载；梯度可经检测损失反传到贴片

```bash
pip install -r requirements.txt
```

> 本环境 torch 为 CPU 版，默认配置为小规模（合成数据 + 轻量检测器），整套 4 攻击 × 3 防御约 3–5 分钟完成。

---

## 二、快速开始

```bash
cd patch_attack_defense

# 1) 准备数据集（默认合成，无需下载；如需 COCO 见下文）
python main.py prepare-data --synthetic

# 2) 列出已注册攻防方法
python main.py list

# 3) 全对阵评测（默认 4 攻击 × 3 防御）
python main.py run

# 4) 单对阵
python main.py matchup -a gradient_patch -d resnet18_classifier

# 5) 加载选手提交跑对阵
python main.py submission -a submissions/attack_template -d submissions/defense_template
```

结果输出到 `results/`：
- `summary.json` — 全部对阵与指标
- `rankings.csv` — 攻击/防御最终排名
- `matrix_attack_score.csv` / `matrix_defense_score.csv` / `matrix_evasion.csv` / `matrix_tpr.csv` — 攻防矩阵
- `attack_details.csv` / `defense_details.csv` — 单方法明细
- `heatmap_attack.png` / `heatmap_defense_tpr.png` / `heatmap.html` — 矩阵热力图

---

## 三、数据集

### 合成数据集（默认，离线，CPU 友好）
```bash
python main.py prepare-data --synthetic
```
生成 `data/synthetic/`（COCO 格式目录 + 训练好的轻量检测器 `detector.pt`）。
含 person/car/bus 目标对象、自然背景渐变 + hard negative（类 logo/条纹方块，防止防御方学到“方块=攻击”）。

### COCO 2017（真实数据，需联网）
```bash
# 子集模式：下载完整标注 + 仅按目标类别下载 N 张图（远小于完整 778MB）
python main.py prepare-data --coco --max-images 200

# 或完整 val2017（~778MB）
python main.py prepare-data --coco --full
```
下载到 `data/coco/`（断点续传 + 重试 + 进度）。完成后在 `config.yaml` 设 `dataset.source: coco`。

> 自动探测：`dataset.source: auto` 时优先用 COCO，缺失则回退合成。

---

## 四、检测器

| 角色 | 模型 | 说明 |
|---|---|---|
| Model-A（公开白盒） | `fasterrcnn_mobilenet_v3_large_fpn` (torchvision) | 攻击方可前向+反传梯度 |
| Model-B（隐藏迁移，可选） | `fasterrcnn_resnet50_fpn` | 架构差异更大，`detectors.model_b.enabled: true` 开启 |
| 合成模式 Model-A | `TinyDetector`（内置轻量单阶段检测器） | 无网络时自动训练，可微，供攻击反传 |

> 赛题“建议”用 YOLO 作为 Model-A；本环境无 ultralytics 且 CPU 下 torchvision 更稳，故用 torchvision 预训练检测器替代（经 `Detector` 抽象类封装，YOLO 可按同接口插入）。

---

## 五、攻防解耦机制（核心）

### 攻击接口（`attacks/`，赛题 §15-§18）
```python
def attack(env, task) -> {"patch": ndarray (3, patch_size, patch_size) float[0,1]}
# task = {images, annotations, target_class, patch_size, max_patch_area_ratio, detector}
```
攻击方**不**返回改图/位置/缩放/旋转。同一轮所有测试样本用同一张贴片（Universal）。

### 防御接口（`defenses/`，赛题 §27-§30）
```python
class DefenseModel:
    def __init__(self, model_path): ...
    def predict(self, image) -> float   # patch_probability in [0,1]
# 或 def defend(env, images) -> ndarray[N]
```
防御方只接收 image，不获框/类别/贴片/检测预测，不访问检测器（`DefenseEnv` 不暴露检测器）。

### 裁判统一渲染（`judge/patch_renderer.py`，赛题 §12/§19，不提供给防御方）
```python
apply_patch(image, target_bbox, patch, rng)   # 贴片中心落目标框中央 60%；旋转±15°；缩放 0.85-1.15；颜色抖动
apply_occlusion(image, bbox, params, rng)     # 相同参数渲染 gray/black/white/noise/texture 自然遮挡基线
```
`AttackGain` 用相同参数的自然遮挡作基线，公平扣减“纯遮挡”影响（赛题 §4.4/§23）。

### 自动发现（`judge/registry.py`）
内置 `attacks/`、`defenses/` 自动注册；同时支持类（继承 Base）与裸函数（自动包装）；
外部选手提交目录（`attack.py`/`defense.py`）通过 `main.py submission` 加载。

---

## 六、内置攻防池

**攻击 A0**（`attacks/`）：
| 名称 | 类型 | 说明 |
|---|---|---|
| `random_noise` | A0-1 基线 | 随机噪声贴片 |
| `checkerboard` | A0-2 基线 | 棋盘纹理贴片 |
| `gray_patch` | 自然遮挡对照 | gray/black/white（AttackGain 预期≈0，验证遮挡扣减） |
| `gradient_patch` | 梯度优化 | DPatch + EOT + TV 正则，最大化检测损失隐藏目标 |

**防御 D0**（`defenses/`）：
| 名称 | 类型 | 说明 |
|---|---|---|
| `resnet18_classifier` | D0-1 | ResNet18 二分类（ImageNet 预训练） |
| `mobilenetv3_classifier` | D0-2 | MobileNetV3-Large 二分类 |
| `frequency_classifier` | 频域 | FFT 幅度谱特征 + 逻辑回归（无大权重，极快） |

---

## 七、评测指标（严格对齐赛题 §10/§20-§42）

**攻击**：`Recall_clean/adv/occ`、`HideRate=1-Recall`、
`AttackGain=max(0,(Hide_adv-Hide_occ)/(1-Hide_occ+ε))`、`ConfidenceDrop`、
`AttackScore_base=0.85·Gain+0.15·ConfDrop`

**防御**：`TPR`、`FPR`、`TNR`、`BalancedAccuracy=(TPR+TNR)/2`、`AUROC`、
`DefenseScore=0.7·BalAcc+0.3·AUROC`

**全对阵**：`EvasionRate=1-TPR`、
`AttackScore_{i,j}=AttackScore_base·(0.8+0.2·Evasion)`、
`FinalAttackScore=mean over D_q`、`FinalDefenseScore=mean over A_q`

**资格线**：攻击 `AttackGain≥0.40 & HideRate_adv≥0.50`；防御 `FPR≤10% & TPR_baseline≥70%`

> 合成轻量检测器上攻击 AttackGain 较小（检测器对部分贴片较鲁棒），主要用于验证流水线与解耦；切换到真实 COCO + torchvision 检测器后攻击效果显著增强。

---

## 八、目录结构

```
patch_attack_defense/
├── main.py                      # CLI: prepare-data/list/run/matchup/submission
├── config.yaml                  # 主配置
├── judge/                       # 裁判程序
│   ├── config.py env.py dataset.py detector.py tiny_detector.py
│   ├── patch_renderer.py attack_validator.py defense_runner.py
│   ├── metrics.py registry.py evaluator.py reporting.py utils.py
├── attacks/                     # 内置攻击池 A0
├── defenses/                    # 内置防御池 D0
├── data/                        # download_coco.py / make_synthetic_dataset.py / coco/ / synthetic/
├── submissions/                 # 选手提交模板（attack_template / defense_template）
└── results/                     # 运行产物
```

---

## 九、扩展自定义攻防

### 新增攻击
在 `attacks/` 下新建 `my_attack.py`：
```python
from .base import BaseAttack
class MyAttack(BaseAttack):
    name = "my_attack"
    def __call__(self, env, task):
        # 用 env.detector (predict/loss/hiding_loss) 与 env.render_patch_torch (可微渲染, EOT)
        ...
        return {"patch": patch}   # (3, patch_size, patch_size) float[0,1]
```
自动注册，`python main.py run --attacks my_attack` 即可。

### 新增防御
在 `defenses/` 下新建 `my_defense.py`：
```python
from .base import BaseDefense, DefenseModel
class MyDefense(BaseDefense):
    name = "my_defense"
    def fit(self, denv, clean_images, patched_images, cfg):
        # 训练 -> 返回 DefenseModel 实例 (predict(image)->float)
        ...
        return model
```
`python main.py run --defenses my_defense`。

### 选手提交
按 `submissions/attack_template/`、`submissions/defense_template/` 目录规范填写 `attack.py`/`defense.py`，`python main.py submission -a <dir> -d <dir>` 加载评测。

---

## 十、Python API

```python
from judge.evaluator import Evaluator
from judge import registry
registry.discover_builtin()
ev = Evaluator.from_config("config.yaml")
res = ev.run_all_vs_all()   # 返回结果 dict
ev.save("results")
print(res["rankings"])
```
