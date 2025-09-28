import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.TransCC import (
    Encoder,
    CBAMResidualBlock,
    SkipConnection,
    FeatureAggregator,
    ConvBlock,
)


class DecoderV2(nn.Module):
    """TransCC V2 decoder with multi-scale supervision and boundary heads."""

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        img_size: int = 512,
        num_classes: int = 2,
        decoder_channels: List[int] = [512, 256, 128, 64],
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels

        # 将 transformer 特征映射到卷积特征空间
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, decoder_channels[0]),
            nn.LayerNorm(decoder_channels[0])
        )

        # 初始卷积和注意力块
        self.initial_conv = ConvBlock(decoder_channels[0], decoder_channels[0])

        self.cbam_blocks = nn.ModuleList([
            CBAMResidualBlock(decoder_channels[0], decoder_channels[0]),
            CBAMResidualBlock(decoder_channels[1], decoder_channels[1]),
            CBAMResidualBlock(decoder_channels[2], decoder_channels[2])
        ])

        # 逐层上采样
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3], kernel_size=4, stride=2, padding=1, bias=False)
        ])

        self.upsample_bn = nn.ModuleList([
            nn.BatchNorm2d(decoder_channels[1]),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.BatchNorm2d(decoder_channels[3])
        ])

        # 跳级连接与特征聚合
        self.skip_connections = nn.ModuleList([
            SkipConnection(embed_dim, decoder_channels[1], decoder_channels[1]),
            SkipConnection(embed_dim, decoder_channels[2], decoder_channels[2]),
            SkipConnection(embed_dim, decoder_channels[3], decoder_channels[3])
        ])

        self.feature_aggregators = nn.ModuleList([
            FeatureAggregator(decoder_channels[1], decoder_channels[1]),
            FeatureAggregator(decoder_channels[2], decoder_channels[2]),
            FeatureAggregator(decoder_channels[3], decoder_channels[3])
        ])

        # 多尺度分割头
        self.seg_head_stage1 = nn.Sequential(
            ConvBlock(decoder_channels[1], decoder_channels[1] // 2),
            nn.Conv2d(decoder_channels[1] // 2, num_classes, kernel_size=1)
        )
        self.seg_head_stage2 = nn.Sequential(
            ConvBlock(decoder_channels[2], decoder_channels[2] // 2),
            nn.Conv2d(decoder_channels[2] // 2, num_classes, kernel_size=1)
        )
        self.final_proj = nn.Sequential(
            ConvBlock(decoder_channels[3], decoder_channels[3] // 2),
            ConvBlock(decoder_channels[3] // 2, decoder_channels[3] // 2)
        )
        self.seg_head_main = nn.Conv2d(decoder_channels[3] // 2, num_classes, kernel_size=1)

        # 边界监督分支（输入应为128通道）
        self.boundary_head_aux = nn.Sequential(
            ConvBlock(decoder_channels[2], decoder_channels[2] // 2),
            nn.Conv2d(decoder_channels[2] // 2, 1, kernel_size=1)
        )
        self.boundary_head_main = nn.Sequential(
            nn.Conv2d(decoder_channels[3] // 2, decoder_channels[3] // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3] // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3] // 4, 1, kernel_size=1)
        )

        self.dropout = nn.Dropout(p=0.1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        encoder_features: torch.Tensor,
        layer_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning main seg, boundary, and auxiliary outputs."""
        B, N_plus_1, _ = encoder_features.shape
        patch_features = encoder_features[:, 1:, :]
        features = self.feature_proj(patch_features)

        H = W = int(features.shape[1] ** 0.5)
        x = features.transpose(1, 2).reshape(B, self.decoder_channels[0], H, W)
        x = self.initial_conv(x)

        # Stage 1: 32 -> 64
        x = self.cbam_blocks[0](x)
        x = F.relu(self.upsample_bn[0](self.upsample_layers[0](x)))
        if len(layer_features) >= 1:
            x = self.skip_connections[0](layer_features[0], x)
        x = self.feature_aggregators[0](x)
        seg_aux1 = self.seg_head_stage1(x)

        # Stage 2: 64 -> 128
        x = self.cbam_blocks[1](x)
        x = F.relu(self.upsample_bn[1](self.upsample_layers[1](x)))
        if len(layer_features) >= 2:
            x = self.skip_connections[1](layer_features[1], x)
        x = self.feature_aggregators[1](x)
        seg_aux2 = self.seg_head_stage2(x)


        # Stage 3: 128 -> 256
        x = self.cbam_blocks[2](x)
        x_128 = x  # 128通道特征（上采样前）
        boundary_aux = self.boundary_head_aux(x_128)
        x = F.relu(self.upsample_bn[2](self.upsample_layers[2](x)))
        if len(layer_features) >= 3:
            x = self.skip_connections[2](layer_features[2], x)
        x = self.feature_aggregators[2](x)

        # Final stage: 256 -> 512
        x = F.relu(self.upsample_bn[3](self.upsample_layers[3](x)))
        x = self.dropout(x)
        x = self.final_proj(x)
        seg_main = self.seg_head_main(x)
        boundary_main = self.boundary_head_main(x)

        # 将辅助输出上采样到原图尺寸
        seg_aux1 = F.interpolate(seg_aux1, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        seg_aux2 = F.interpolate(seg_aux2, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        boundary_aux = F.interpolate(boundary_aux, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        boundary_main = F.interpolate(boundary_main, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return seg_main, boundary_main, seg_aux1, seg_aux2, boundary_aux


class TransCC_V2(nn.Module):
    """TransCC Version 2 with boundary-aware multi-scale decoder."""

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        decoder_channels: List[int] = [512, 256, 128, 64],
        use_fourier_pos: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_fourier_pos=use_fourier_pos,
        )
        self.decoder = DecoderV2(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_features, layer_features = self.encoder(x)
        outputs = self.decoder(encoder_features, layer_features)
        return outputs


def create_transcc_v2(config: Optional[dict] = None) -> TransCC_V2:
    if config is None:
        config = {
            'img_size': 512,
            'patch_size': 16,
            'in_chans': 3,
            'num_classes': 2,
            'embed_dim': 768,
            'depth': 6,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'decoder_channels': [512, 256, 128, 64],
            'use_fourier_pos': True,
        }
    return TransCC_V2(**config)
