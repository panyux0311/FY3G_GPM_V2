"""V6 U-Net backbone with a parameterized three-lead, surface-only output contract."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from schema import PATCH_SIZE
from utils.config import ModelConfig

from .precipitation import PrecipitationModel

SPATIAL_SIZE = PATCH_SIZE


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.refine = DoubleConv(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat((skip, x), dim=1))


class UNetStandard(PrecipitationModel):
    architecture_name = "unet_standard"

    def __init__(self, config: ModelConfig) -> None:
        width = int(config.base_width)
        super().__init__(config, decoded_channels=width, feature_channels=width * 4)
        self.enc1 = DoubleConv(self.input_steps * self.input_channels, width)
        self.enc2 = DoubleConv(width, width * 2)
        self.enc3 = DoubleConv(width * 2, width * 4)
        self.enc4 = DoubleConv(width * 4, width * 8)
        self.bottleneck = DoubleConv(width * 8, width * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = UNetUp(width * 16, width * 8)
        self.up3 = UNetUp(width * 8, width * 4)
        self.up2 = UNetUp(width * 4, width * 2)
        self.up1 = UNetUp(width * 2, width)

    def _validate_input(self, value: torch.Tensor) -> torch.Tensor:
        expected = (self.input_steps, self.input_channels, SPATIAL_SIZE, SPATIAL_SIZE)
        if value.ndim != 5 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"model input must be [B,{','.join(map(str, expected))}], got {tuple(value.shape)}")
        return value.reshape(value.shape[0], self.input_steps * self.input_channels, SPATIAL_SIZE, SPATIAL_SIZE)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5 or x.shape[2] < 40:
            raise ValueError("U-Net input must expose 20 reflectivity-valid channels")
        flat = self._validate_input(x)
        e1 = self.enc1(flat)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        decoded = self.up1(self.up2(self.up3(self.up4(bottleneck, e4), e3), e2), e1)
        return self._forecast_outputs(x, decoded, e3)


