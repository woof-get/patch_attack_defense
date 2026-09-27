"""D0-2 MobileNetV3 二分类防御 (轻量化参考, 赛题 §46)。"""
from .base import BaseDefense, train_cnn_binary_classifier


class MobileNetV3Defense(BaseDefense):
    name = "mobilenetv3_classifier"
    meta = {"type": "learned", "reference": "D0-2"}

    def fit(self, denv, clean_images, patched_images, cfg):
        tcfg = getattr(cfg, "training", None)
        dcfg = getattr(tcfg, "defense", None) if tcfg else None

        def builder(pretrained=True):
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            w = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            return mobilenet_v3_large(weights=w)

        return train_cnn_binary_classifier(
            name=self.name, backbone_builder=builder,
            clean_images=clean_images, patched_images=patched_images,
            device=denv.device,
            epochs=int(getattr(dcfg, "epochs", 5)),
            batch_size=int(getattr(dcfg, "batch_size", 32)),
            lr=float(getattr(dcfg, "lr", 1e-3)),
            pretrained=bool(getattr(dcfg, "backbone_pretrained", True)),
            seed=int(getattr(getattr(cfg, "run", None), "seed", 42)),
        )
