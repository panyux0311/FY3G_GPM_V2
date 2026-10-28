"""SimVPv2-style spatial encoder, gSTA temporal translator, and decoder.

The architecture follows Tan et al., SimVPv2 (arXiv:2211.12509). This
implementation is written for the project's three-frame precipitation contract.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from schema import PATCH_SIZE
from utils.config import ModelConfig

from .precipitation import PrecipitationModel


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1.0 - self.probability
        mask = torch.rand((x.shape[0], 1, 1, 1), device=x.device) < keep
        return x * mask.to(x.dtype) / keep


class GatedSpatialBlock(nn.Module):
    """Large-kernel gated spatial attention plus a depthwise convolution MLP."""

    def __init__(self, channels: int, mlp_ratio: float, drop_path: float) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.norm1 = nn.BatchNorm2d(channels)
        self.project_in = nn.Conv2d(channels, channels, 1)
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dilated = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels)
        self.gate = nn.Conv2d(channels, channels * 2, 1)
        self.project_out = nn.Conv2d(channels, channels, 1)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.layer_scale_1 = nn.Parameter(torch.full((channels, 1, 1), 1e-2))
        self.layer_scale_2 = nn.Parameter(torch.full((channels, 1, 1), 1e-2))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.dilated(self.local(F.gelu(self.project_in(self.norm1(x)))))
        value, gate = self.gate(attention).chunk(2, dim=1)
        x = x + self.drop_path(self.layer_scale_1 * self.project_out(value * gate.sigmoid()))
        return x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))


class SpatialBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, downsample: bool = False) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=2 if downsample else 1, padding=1),
            nn.GroupNorm(1, output_channels),
            nn.SiLU(),
        )


class SimVPv2(PrecipitationModel):
    architecture_name = "simvpv2"

    def __init__(self, config: ModelConfig) -> None:
        spatial = int(config.simvpv2_hidden_spatial)
        temporal = int(config.simvpv2_hidden_temporal)
        stages = int(config.simvpv2_spatial_layers)
        blocks = int(config.simvpv2_temporal_layers)
        if spatial <= 0 or temporal <= 0 or stages < 2 or stages % 2 or blocks < 2:
            raise ValueError("SimVPv2 requires positive widths, even spatial layers >= 2, and temporal layers >= 2")
        super().__init__(config, decoded_channels=spatial, feature_channels=spatial)

        downsample = [stage % 2 == 1 for stage in range(stages)]
        self.encoder = nn.ModuleList(
            SpatialBlock(self.input_channels if stage == 0 else spatial, spatial, downsample[stage])
            for stage in range(stages)
        )
        self.temporal_in = nn.Conv2d(self.input_steps * spatial, temporal, 1)
        self.temporal_blocks = nn.Sequential(*[
            GatedSpatialBlock(
                temporal, config.simvpv2_mlp_ratio,
                config.simvpv2_drop_path * stage / max(1, blocks - 1),
            )
            for stage in range(blocks)
        ])
        self.temporal_out = nn.Conv2d(temporal, self.input_steps * spatial, 1)
        if self.multi_lead:
            self.skip_fusion = nn.Conv2d(self.input_steps * spatial, self.forecast_leads * spatial, 1, bias=False)
        self.decoder = nn.ModuleList(SpatialBlock(spatial, spatial) for _ in range(stages))
        self.upsample = list(reversed(downsample))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        expected = (self.input_steps, self.input_channels, PATCH_SIZE, PATCH_SIZE)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"SimVPv2 input must be [B,{','.join(map(str, expected))}], got {tuple(x.shape)}")
        batch, steps, channels, height, width = x.shape
        encoded = x.reshape(batch * steps, channels, height, width)
        skip = None
        for stage, layer in enumerate(self.encoder):
            encoded = layer(encoded)
            if stage == 0:
                frames = encoded.reshape(batch, steps, *encoded.shape[1:])
                if self.multi_lead:
                    observed = x[:, :, 20:40].bool().any(dim=2, keepdim=True)
                    skip = self.skip_fusion((frames * observed).reshape(batch, -1, height, width))
                    skip = skip.reshape(batch * self.forecast_leads, *frames.shape[2:])
                else:
                    skip = frames[:, -1]

        latent_shape = encoded.shape[1:]
        stacked = encoded.reshape(batch, steps * latent_shape[0], *latent_shape[1:])
        translated = self.temporal_out(self.temporal_blocks(self.temporal_in(stacked)))
        features = translated.reshape(batch, self.forecast_leads, *latent_shape)
        feature = features if self.multi_lead else features[:, -1]

        decoded = feature.reshape(batch * self.forecast_leads, *latent_shape) if self.multi_lead else feature
        for stage, layer in enumerate(self.decoder):
            if self.upsample[stage]:
                decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
            if stage == len(self.decoder) - 1:
                decoded = decoded + skip
            decoded = layer(decoded)
        if self.multi_lead:
            decoded = decoded.reshape(batch, self.forecast_leads, *decoded.shape[1:])
        return self._forecast_outputs(x, decoded, feature)
