"""梯度优化对抗贴片攻击 (DPatch 思想, 赛题 §5.2/§54)。

- 用公开检测器 Model-A 的训练损失做梯度上升 (最大化检测损失 = 隐藏目标)。
- EOT: 每步随机采样贴片位置/缩放/旋转 (经 env.render_patch_torch 可微渲染)。
- TV 正则 (平滑/可打印性)。
- 计算图: patch -> 渲染图 -> 检测器 -> loss, 反传到 patch。
"""
import numpy as np

from .base import BaseAttack


def _total_variation(p):
    """p: (3,H,W) tensor。返回标量 TV。"""
    return (p[:, 1:, :] - p[:, :-1, :]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()


class GradientPatchAttack(BaseAttack):
    name = "gradient_patch"
    meta = {"type": "optimization", "reference": "DPatch/EOT"}

    def __init__(self, iters=20, lr=0.03, batch_size=4, tv_weight=0.01, init="noise", seed=42):
        self.iters = iters
        self.lr = lr
        self.batch_size = batch_size
        self.tv_weight = tv_weight
        self.init = init
        self.seed = seed

    def __call__(self, env, task):
        import torch
        from .base import BaseAttack
        rng = np.random.default_rng(self.seed)
        ps = int(task["patch_size"])
        device = env.device
        images = task["images"]
        annotations = task["annotations"]
        target_class = int(task["target_class"])

        # 初始化贴片
        p_np = BaseAttack.init_patch(ps, rng, mode=self.init)
        patch = torch.from_numpy(p_np).to(device).float().requires_grad_(True)
        opt = torch.optim.Adam([patch], lr=self.lr)

        # 预筛: 含目标类对象的训练图
        idxs = [i for i, anns in enumerate(annotations)
                if any(int(a["category_id"]) == target_class for a in anns)]
        if not idxs:
            return {"patch": patch.detach().cpu().numpy()}
        idxs_arr = np.array(idxs)

        iters = self.iters
        bs = min(self.batch_size, len(idxs))
        det = env.detector
        log_every = max(1, iters // 5)

        for it in range(iters):
            batch = rng.choice(idxs_arr, size=bs, replace=(len(idxs) < bs))
            rendered_list, tgt_list = [], []
            for i in batch:
                img = images[int(i)]
                anns = [a for a in annotations[int(i)] if int(a["category_id"]) == target_class]
                if not anns:
                    continue
                # EOT: 每图每步随机采样渲染参数
                bbox = anns[int(rng.integers(len(anns)))]["bbox"]
                params = env.sample_params(bbox, rng)
                img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
                rendered = env.render_patch_torch(img_t, patch, params)[0]  # (3,H,W) 连接 patch
                rendered_list.append(rendered)
                boxes = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32, device=device)
                labels = torch.tensor([target_class] * len(anns), dtype=torch.long, device=device)
                tgt_list.append({"boxes": boxes, "labels": labels})

            if not rendered_list:
                continue
            tv = _total_variation(patch)
            # 隐藏目标: 最小化 detector.hiding_loss (直接降 objectness + 扰乱框, 非饱和)
            if hasattr(det, "hiding_loss"):
                _, hide_loss = det.hiding_loss(rendered_list, tgt_list)
                objective = hide_loss + self.tv_weight * tv  # 最小化
            else:
                _, losses = det.loss(rendered_list, tgt_list)
                hide = sum(v for k, v in losses.items() if "box" not in k.lower())
                objective = -hide + self.tv_weight * tv
            opt.zero_grad()
            objective.backward()
            # 防 NaN
            if patch.grad is not None:
                patch.grad = torch.nan_to_num(patch.grad, nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()
            with torch.no_grad():
                patch.clamp_(0.0, 1.0)
            if (it + 1) % log_every == 0 or it == 0:
                from judge.utils import log
                log.info(f"[gradient_patch] iter {it+1}/{iters}  obj={float(objective.detach()):.4f}  tv={float(tv.detach()):.4f}")

        return {"patch": patch.detach().cpu().numpy().astype(np.float32)}
