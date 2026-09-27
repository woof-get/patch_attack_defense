"""D0-1 ResNet18 二分类防御 (赛题 §46/§55)。"""
from .base import BaseDefense, train_cnn_binary_classifier


class ResNet18Defense(BaseDefense):
    name = "resnet18_classifier"
    meta = {"type": "learned", "reference": "D0-1"}

    def fit(self, denv, clean_images, patched_images, cfg):
        tcfg = getattr(cfg, "training", None)
        dcfg = getattr(tcfg, "defense", None) if tcfg else None

        def builder(pretrained=True):
            from torchvision.models import resnet18, ResNet18_Weights
            w = ResNet18_Weights.DEFAULT if pretrained else None
            return resnet18(weights=w)

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
