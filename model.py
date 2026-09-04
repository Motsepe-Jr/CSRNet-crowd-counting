"""CSRNet architecture: a VGG16-frontend dilated CNN for crowd density."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class CSRNet(nn.Module):
    """Dilated CNN that regresses a density map from an RGB image."""

    def __init__(self, load_weights: bool = False) -> None:
        super().__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Initialise *all* layers first, then overwrite the frontend with VGG16
        # weights so the pretrained features survive.
        self._initialize_weights()
        if not load_weights:
            mod = vgg16(weights=VGG16_Weights.DEFAULT)
            # Copy the conv weights of the first 13 conv layers / 23 modules
            # of vgg16.features onto our frontend (same ordering).
            self._copy_vgg_frontend(mod)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _copy_vgg_frontend(self, vgg_model: nn.Module) -> None:
        """Copy conv weights/biases from a VGG16 ``features`` module."""
        src_convs = [m for m in vgg_model.features if isinstance(m, nn.Conv2d)]
        dst_convs = [m for m in self.frontend if isinstance(m, nn.Conv2d)]
        n = min(len(src_convs), len(dst_convs))
        for src, dst in zip(src_convs[:n], dst_convs[:n]):
            dst.weight.data.copy_(src.weight.data)
            if dst.bias is not None and src.bias is not None:
                dst.bias.data.copy_(src.bias.data)


def make_layers(
    cfg,
    in_channels: int = 3,
    batch_norm: bool = False,
    dilation: bool = False,
) -> nn.Sequential:
    d_rate = 2 if dilation else 1
    layers: list[nn.Module] = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
