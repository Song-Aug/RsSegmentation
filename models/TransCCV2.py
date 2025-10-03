import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBlock(nn.Module):
    """基础卷积块：Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """CBAM 中的通道注意力模块"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM 中的空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CBAMResidualBlock(nn.Module):
    """带有 CBAM 的残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(CBAMResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.cbam = CBAM(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class FourierPositionEmbedding(nn.Module):
    def __init__(self, embed_dim, max_freq=10000):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        self.freq_dim = embed_dim // 4
        freq_bands = torch.exp(torch.linspace(0, math.log(max_freq), self.freq_dim))
        self.register_buffer('freq_bands', freq_bands)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, patch_centers):
        x_coords = patch_centers[:, :, 0:1]
        y_coords = patch_centers[:, :, 1:2]
        x_freqs = x_coords * self.freq_bands.unsqueeze(0).unsqueeze(0)
        y_freqs = y_coords * self.freq_bands.unsqueeze(0).unsqueeze(0)
        fourier_features = torch.cat([torch.sin(x_freqs), torch.cos(x_freqs), torch.sin(y_freqs), torch.cos(y_freqs)], dim=-1)
        return self.proj(fourier_features)
    
    def get_patch_centers(self, img_size, patch_size):
        img_h, img_w = to_2tuple(img_size)
        patch_h, patch_w = to_2tuple(patch_size)
        num_patches_h = img_h // patch_h
        num_patches_w = img_w // patch_w
        y_centers = (torch.arange(num_patches_h, dtype=torch.float32) + 0.5) * patch_h / img_h
        x_centers = (torch.arange(num_patches_w, dtype=torch.float32) + 0.5) * patch_w / img_w
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing='ij')
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


class Encoder(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768, depth=6,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_fourier_pos=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_fourier_pos = use_fourier_pos
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if use_fourier_pos:
            self.pos_embed = FourierPositionEmbedding(embed_dim)
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            num_patches = (img_size // patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.use_fourier_pos:
            patch_centers = self.pos_embed.get_patch_centers((H, W), self.patch_size).to(x.device)
            patch_centers = patch_centers.unsqueeze(0).expand(B, -1, -1)
            fourier_pos = self.pos_embed(patch_centers)
            x[:, 1:, :] += fourier_pos
            x[:, 0:1, :] += self.cls_pos_embed
        else:
            x += self.pos_embed
        
        x = self.pos_drop(x)
        layer_features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [1, 3, 5]:  # Collect features from specific layers
                layer_features.append(x[:, 1:, :])
        
        return self.norm(x), layer_features


class SkipConnection(nn.Module):
    """跳级连接模块，用于融合编码器和解码器特征"""
    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super(SkipConnection, self).__init__()
        self.encoder_proj = nn.Sequential(
            nn.Linear(encoder_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        self.decoder_conv = ConvBlock(decoder_channels, out_channels, kernel_size=1, padding=0)
        self.fusion_conv = ConvBlock(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.attention = CBAM(out_channels)
    
    def forward(self, encoder_feat, decoder_feat):
        B, N, _ = encoder_feat.shape
        _, _, H, W = decoder_feat.shape
        encoder_proj = self.encoder_proj(encoder_feat)
        encoder_2d = encoder_proj.transpose(1, 2).reshape(B, -1, int(N**0.5), int(N**0.5))
        encoder_upsampled = F.interpolate(encoder_2d, size=(H, W), mode='bilinear', align_corners=False)
        decoder_proj = self.decoder_conv(decoder_feat)
        fused = torch.cat([encoder_upsampled, decoder_proj], dim=1)
        fused = self.fusion_conv(fused)
        return self.attention(fused)


class FeatureAggregator(nn.Module):
    """多尺度特征聚合模块"""
    def __init__(self, in_channels, out_channels):
        super(FeatureAggregator, self).__init__()
        self.conv1x1 = ConvBlock(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv3x3 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        self.attention = CBAM(out_channels)
    
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.attention(x)
        return x

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module."""
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ])
        for rate in rates:
            self.aspp_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        size = x.shape[2:]
        features = [conv(x) for conv in self.aspp_convs]
        pooled_features = self.global_avg_pool(x)
        pooled_features = F.interpolate(pooled_features, size=size, mode='bilinear', align_corners=False)
        features.append(pooled_features)
        x = torch.cat(features, dim=1)
        x = self.conv_out(x)
        return self.dropout(x)


class DecoderV2(nn.Module):
    """TransCC V2 decoder with multi-scale supervision and boundary heads."""
    def __init__(self, embed_dim: int = 768, patch_size: int = 16, img_size: int = 512,
                 num_classes: int = 2, decoder_channels: List[int] = [512, 256, 128, 64]):
        super().__init__()
        self.img_size = img_size
        self.decoder_channels = decoder_channels
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, decoder_channels[0]),
            nn.LayerNorm(decoder_channels[0])
        )
        self.initial_conv = ConvBlock(decoder_channels[0], decoder_channels[0])
        self.cbam_blocks = nn.ModuleList([
            CBAMResidualBlock(decoder_channels[0], decoder_channels[0]),
            CBAMResidualBlock(decoder_channels[1], decoder_channels[1]),
            CBAMResidualBlock(decoder_channels[2], decoder_channels[2])
        ])
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(decoder_channels[0], decoder_channels[1])),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(decoder_channels[1], decoder_channels[2])),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(decoder_channels[2], decoder_channels[3])),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(decoder_channels[3], decoder_channels[3]))
        ])
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
        self.seg_head_stage1 = nn.Sequential(ConvBlock(decoder_channels[1], decoder_channels[1] // 2), nn.Conv2d(decoder_channels[1] // 2, num_classes, 1))
        self.seg_head_stage2 = nn.Sequential(ConvBlock(decoder_channels[2], decoder_channels[2] // 2), nn.Conv2d(decoder_channels[2] // 2, num_classes, 1))
        self.final_proj = ASPP(decoder_channels[3], decoder_channels[3] // 2)
        self.seg_head_main = nn.Conv2d(decoder_channels[3] // 2, num_classes, 1)
        self.boundary_head_aux = nn.Sequential(ConvBlock(decoder_channels[2], decoder_channels[2] // 2), nn.Conv2d(decoder_channels[2] // 2, 1, 1))
        
        # --- 强化的边界预测头 ---
        self.boundary_head_main = nn.Sequential(
            ConvBlock(decoder_channels[3] // 2, decoder_channels[3] // 4, kernel_size=3),
            CBAM(decoder_channels[3] // 4),
            ConvBlock(decoder_channels[3] // 4, decoder_channels[3] // 8, kernel_size=3),
            nn.Conv2d(decoder_channels[3] // 8, 1, 1)
        )

        self.dropout = nn.Dropout(p=0.1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, encoder_features, layer_features):
        B, _, _ = encoder_features.shape
        patch_features = encoder_features[:, 1:, :]
        features = self.feature_proj(patch_features)
        H = W = int(features.shape[1] ** 0.5)
        x = features.transpose(1, 2).reshape(B, self.decoder_channels[0], H, W)
        x = self.initial_conv(x)
        
        # Stage 1
        x = self.cbam_blocks[0](x)
        x = self.upsample_layers[0](x)
        if len(layer_features) >= 1:
            x = self.skip_connections[0](layer_features[0], x)
        x = self.feature_aggregators[0](x)
        seg_aux1 = self.seg_head_stage1(x)

        # Stage 2
        x = self.cbam_blocks[1](x)
        x = self.upsample_layers[1](x)
        if len(layer_features) >= 2:
            x = self.skip_connections[1](layer_features[1], x)
        x = self.feature_aggregators[1](x)
        seg_aux2 = self.seg_head_stage2(x)

        # Stage 3
        x = self.cbam_blocks[2](x)
        boundary_aux = self.boundary_head_aux(x)
        x = self.upsample_layers[2](x)
        if len(layer_features) >= 3:
            x = self.skip_connections[2](layer_features[2], x)
        x = self.feature_aggregators[2](x)

        # Final stage
        x = self.upsample_layers[3](x)
        x = self.dropout(x)
        x = self.final_proj(x)
        seg_main = self.seg_head_main(x)
        boundary_main = self.boundary_head_main(x)

        # Upsample all outputs to original image size for loss calculation
        seg_main_out = F.interpolate(seg_main, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        seg_aux1_out = F.interpolate(seg_aux1, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        seg_aux2_out = F.interpolate(seg_aux2, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        boundary_aux_out = F.interpolate(boundary_aux, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        boundary_main_out = F.interpolate(boundary_main, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return seg_main_out, boundary_main_out, seg_aux1_out, seg_aux2_out, boundary_aux_out


class TransCC_V2(nn.Module):
    """TransCC Version 2 with boundary-aware multi-scale decoder."""
    def __init__(self, img_size: int = 512, patch_size: int = 16, in_chans: int = 3, num_classes: int = 2,
                 embed_dim: int = 768, depth: int = 6, num_heads: int = 12, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1, decoder_channels: List[int] = [512, 256, 128, 64],
                 use_fourier_pos: bool = True):
        super().__init__()
        self.encoder = Encoder(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
            drop_rate, attn_drop_rate, drop_path_rate, use_fourier_pos=use_fourier_pos
        )
        self.decoder = DecoderV2(
            embed_dim, patch_size, img_size, num_classes, decoder_channels
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        encoder_features, layer_features = self.encoder(x)
        return self.decoder(encoder_features, layer_features)


def create_transcc_v2(config: Optional[dict] = None) -> TransCC_V2:
    if config is None:
        config = {
            'img_size': 512, 'patch_size': 16, 'in_chans': 3, 'num_classes': 2, 'embed_dim': 768,
            'depth': 6, 'num_heads': 12, 'mlp_ratio': 4.0, 'qkv_bias': True, 'drop_rate': 0.0,
            'attn_drop_rate': 0.0, 'drop_path_rate': 0.1, 'decoder_channels': [512, 256, 128, 64],
            'use_fourier_pos': True,
        }
    return TransCC_V2(**config)