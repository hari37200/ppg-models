"""The four image classifiers compared in the paper.

    AlexNet, ResNet-50, VGG-16, and the paper's contribution AvgPool_VGG-16.

AvgPool_VGG-16 is plain VGG-16 with every ``MaxPool2d`` in the feature stack
replaced by an ``AvgPool2d`` of identical geometry. The paper's argument is that
average pooling preserves temporal information and suppresses noise, whereas max
pooling keeps only peak values -- which matters when the input is a normalised
waveform drawing rather than a natural image.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PAPER_MODELS", "build_model", "avgpool_vgg16", "replace_maxpool_with_avgpool"]

PAPER_MODELS = ("alexnet", "resnet50", "vgg16", "avgpool_vgg16")


def _load_backbone(name: str, pretrained: bool):
    """Fetch a torchvision backbone across old and new weight APIs."""
    import torchvision.models as tvm

    factories = {
        "alexnet": (tvm.alexnet, "AlexNet_Weights"),
        "resnet50": (tvm.resnet50, "ResNet50_Weights"),
        "vgg16": (tvm.vgg16, "VGG16_Weights"),
    }
    if name not in factories:
        raise ValueError(f"unknown backbone {name!r}; expected one of {list(factories)}")
    factory, weights_enum = factories[name]

    if not pretrained:
        return factory(weights=None) if _has_weights_arg(factory) else factory(pretrained=False)

    if _has_weights_arg(factory):
        weights = getattr(tvm, weights_enum).IMAGENET1K_V1
        return factory(weights=weights)
    return factory(pretrained=True)  # torchvision < 0.13


def _has_weights_arg(factory) -> bool:
    import inspect

    try:
        return "weights" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return True


def replace_maxpool_with_avgpool(module: nn.Module) -> int:
    """Recursively swap every ``MaxPool2d`` for a geometry-matched ``AvgPool2d``.

    Returns the number of layers replaced. Mutates ``module`` in place.
    """
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.MaxPool2d):
            setattr(
                module,
                name,
                nn.AvgPool2d(
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    ceil_mode=child.ceil_mode,
                ),
            )
            n += 1
        else:
            n += replace_maxpool_with_avgpool(child)
    return n


def _reset_classifier(model: nn.Module, name: str, num_classes: int, dropout: float) -> None:
    """Point the final layer at our 4 classes."""
    if name == "resnet50":
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return

    # alexnet / vgg16 both end in classifier[-1] = Linear
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    # Raise dropout a little: 657 segments against ~130 M VGG parameters.
    for m in model.classifier:
        if isinstance(m, nn.Dropout):
            m.p = dropout


def build_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_features: bool = False,
) -> nn.Module:
    """Instantiate one of :data:`PAPER_MODELS`.

    Parameters
    ----------
    pretrained
        Start from ImageNet weights. The paper states "Pre-trained ResNet and
        VGG-16 models are implemented", so this defaults to True.
    freeze_features
        Freeze the convolutional stack and train only the classifier head.

        Note the asymmetry: for ResNet-50 the head is a single ``fc`` layer
        (0.008 M trainable), but for VGG-16 and AlexNet ``classifier`` is a
        three-layer MLP dominated by a 25088x4096 ``Linear``, so 119 M / 54 M
        parameters still train. Freezing therefore saves far less time on the
        VGG family than the name suggests -- it mainly buys regularisation.
    """
    name = name.lower()
    if name not in PAPER_MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {PAPER_MODELS}")

    backbone_name = "vgg16" if name == "avgpool_vgg16" else name
    model = _load_backbone(backbone_name, pretrained)

    if name == "avgpool_vgg16":
        n_swapped = replace_maxpool_with_avgpool(model.features)
        if n_swapped != 5:
            raise RuntimeError(
                f"expected to replace 5 max-pools in VGG-16, replaced {n_swapped}"
            )

    _reset_classifier(model, backbone_name, num_classes, dropout)

    if freeze_features:
        head = model.fc if backbone_name == "resnet50" else model.classifier
        head_params = {id(p) for p in head.parameters()}
        for p in model.parameters():
            if id(p) not in head_params:
                p.requires_grad = False

    return model


def avgpool_vgg16(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_features: bool = False,
) -> nn.Module:
    """The paper's proposed model: VGG-16 with average pooling throughout."""
    return build_model(
        "avgpool_vgg16",
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_features=freeze_features,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for m in PAPER_MODELS:
        net = build_model(m, pretrained=False)
        out = net(torch.zeros(2, 3, 224, 224))
        tot, tr = count_parameters(net)
        print(f"{m:<16} out={tuple(out.shape)} params={tot/1e6:.1f}M")
