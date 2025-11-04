import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import collections

# --- 全局常量 (来自 HDNet) ---
BN_MOMENTUM = 0.1
ALIGN_CORNERS = True

# --------------------------------------------------------------------------
# 1. 来自 TransCCV2 的模块 (Transformer + 基础卷积块)
# --------------------------------------------------------------------------

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
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
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
        img_size = to_2tuple(img_size); patch_size = to_2tuple(patch_size)
        self.img_size = img_size; self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2); return x

class FourierPositionEmbedding(nn.Module):
    def __init__(self, embed_dim, max_freq=10000):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        self.freq_dim = embed_dim // 4
        freq_bands = torch.exp(torch.linspace(0, math.log(max_freq), self.freq_dim))
        self.register_buffer('freq_bands', freq_bands)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, patch_centers):
        x_coords = patch_centers[:, :, 0:1]; y_coords = patch_centers[:, :, 1:2]
        x_freqs = x_coords * self.freq_bands.unsqueeze(0).unsqueeze(0)
        y_freqs = y_coords * self.freq_bands.unsqueeze(0).unsqueeze(0)
        fourier_features = torch.cat([torch.sin(x_freqs), torch.cos(x_freqs), torch.sin(y_freqs), torch.cos(y_freqs)], dim=-1)
        return self.proj(fourier_features)
    
    def get_patch_centers(self, img_size, patch_size):
        img_h, img_w = to_2tuple(img_size); patch_h, patch_w = to_2tuple(patch_size)
        num_patches_h = img_h // patch_h; num_patches_w = img_w // patch_w
        y_centers = (torch.arange(num_patches_h, dtype=torch.float32) + 0.5) * patch_h / img_h
        x_centers = (torch.arange(num_patches_w, dtype=torch.float32) + 0.5) * patch_w / img_w
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing='ij')
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)

class TransformerEncoder(nn.Module):
    """(来自 TransCCV2) 全局特征提取器"""
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
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

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
        # --- 修改点 ---
        # 只运行所有块，不再收集
        for block in self.blocks:
            x = block(x)
        
        return self.norm(x) # 只返回最终的全局特征

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)); max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True); max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x); return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__(); self.ca = ChannelAttention(in_planes, ratio); self.sa = SpatialAttention(kernel_size)
    def forward(self, x): x = x * self.ca(x); x = x * self.sa(x); return x

class CBAMResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CBAMResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels))
        self.cbam = CBAM(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = self.shortcut(x); out = self.conv1(x); out = self.conv2(out)
        out = self.cbam(out); out += residual; out = self.relu(out)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[3, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))])
        for rate in rates: self.aspp_convs.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        size = x.shape[2:]; features = [conv(x) for conv in self.aspp_convs]
        pooled_features = self.global_avg_pool(x); pooled_features = F.interpolate(pooled_features, size=size, mode='bilinear', align_corners=False)
        features.append(pooled_features); x = torch.cat(features, dim=1)
        x = self.conv_out(x); return self.dropout(x)

# --------------------------------------------------------------------------
# 2. 来自 HDNet 的模块 (CNN骨干)
# --------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, multi_grid=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample; self.stride = stride
    def forward(self, x):
        residual = x; out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual; out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True); self.downsample = downsample; self.stride = stride
    def forward(self, x):
        residual = x; out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual; out = self.relu(out)
        return out

# 注意：HDNet的Stagetwo_decouple和Stagethree_decouple是 *解码器* 的一部分，
# 我们这里只复用它的 *编码器* 结构 (Stem, Layer1, Transition, Stage2, Transition2)
# 我们将HDNet的Stage2和Stage3的 *分支模块* (BasicBlock堆叠) 作为CNN流
class HDNetStage(nn.Module):
    """
    简化的HDNet Stage，用于多尺度特征提取
    (移除了Stagetwo_decouple中的 flow_field 和 Fusion)
    """
    def __init__(self, input_branches, output_branches, c):
        super(HDNetStage, self).__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches
        
        self.branches = nn.ModuleList()
        for i in range(self.input_branches):
            w = c * (2 ** i)
            branch = nn.Sequential(BasicBlock(w, w), BasicBlock(w, w), BasicBlock(w, w), BasicBlock(w, w))
            self.branches.append(branch)
        
        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='bilinear', align_corners=ALIGN_CORNERS)
                        )
                    )
                else:
                    ops = []
                    for k in range(i - j):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fused = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.input_branches):
                y = y + self.fuse_layers[i][j](x[j])
            x_fused.append(self.relu(y))
        return x_fused

class CNNEncoder(nn.Module):
    """(来自 HDNet) 局部特征提取器"""
    def __init__(self, base_channel: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        downsample = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(256, momentum=BN_MOMENTUM))
        self.layer1 = nn.Sequential(Bottleneck(64, 64, downsample=downsample), Bottleneck(256, 64), Bottleneck(256, 64), Bottleneck(256, 64))
        
        self.transition1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Sequential(nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
        ])
        
        # 我们用简化的HDNetStage替换原来的decouple stage
        self.stage2 = HDNetStage(input_branches=2, output_branches=2, c=base_channel)
        
        self.transition2 = nn.ModuleList([
            nn.Identity(), nn.Identity(),
            nn.Sequential(nn.Sequential(nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
        ])

    def forward(self, x):
        # Stem (512 -> 256)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Stage 1 (256)
        x = self.layer1(x)
        
        # Transition 1 (256 -> 256, 128)
        x_list = [trans(x) for trans in self.transition1]
        skip_256 = x_list[0] # (B, 48, 256, 256)
        
        # Stage 2 (256, 128)
        x_list = self.stage2(x_list)
        skip_128 = x_list[1] # (B, 96, 128, 128)
        
        # Transition 2 (256, 128 -> 256, 128, 64)
        x_list_s3 = [self.transition2[i](x_list[i]) if i < 2 else self.transition2[i](x_list[-1]) for i in range(3)]
        skip_64 = x_list_s3[2] # (B, 192, 64, 64)
        
        # 返回解码器所需的多尺度跳层特征
        # [skip_64, skip_128, skip_256]
        return [skip_64, skip_128, skip_256]


# --------------------------------------------------------------------------
# 3. 混合解码器与新模型
# --------------------------------------------------------------------------

class CNNSkipConnection(nn.Module):
    """(新) 跳级连接模块，用于融合 CNN 局部特征和 Decoder 特征"""
    def __init__(self, cnn_channels, decoder_channels, out_channels):
        super(CNNSkipConnection, self).__init__()
        # 1x1 卷积统一 CNN 特征通道
        self.cnn_proj = ConvBlock(cnn_channels, out_channels, kernel_size=1, padding=0)
        # 1x1 卷积统一 Decoder 特征通道
        self.decoder_proj = ConvBlock(decoder_channels, out_channels, kernel_size=1, padding=0)
        # 融合后的卷积
        self.fusion_conv = ConvBlock(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.attention = CBAM(out_channels)
    
    def forward(self, cnn_feat, decoder_feat):
        # 确保空间尺寸一致 (理论上CNN skip应该更大或相等，这里以上采样为准)
        if cnn_feat.shape[2:] != decoder_feat.shape[2:]:
             cnn_feat = F.interpolate(cnn_feat, size=decoder_feat.shape[2:], mode='bilinear', align_corners=False)

        cnn_proj = self.cnn_proj(cnn_feat)
        decoder_proj = self.decoder_proj(decoder_feat)
        
        fused = torch.cat([cnn_proj, decoder_proj], dim=1)
        fused = self.fusion_conv(fused)
        return self.attention(fused)


class HybridDecoder(nn.Module):
    """
    (来自 TransCCV2) 
    修改后的解码器，主干由Transformer驱动，跳层由CNN驱动
    """
    def __init__(self, embed_dim: int = 768, img_size: int = 512, num_classes: int = 2, 
                 decoder_channels: List[int] = [512, 256, 128, 64],
                 cnn_skip_channels: List[int] = [192, 96, 48]): # [c64, c128, c256]
        super().__init__()
        self.img_size = img_size
        self.decoder_channels = decoder_channels
        
        # 1. Transformer 特征投影 (解码器起点)
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, decoder_channels[0]),
            nn.LayerNorm(decoder_channels[0])
        )
        self.initial_conv = ConvBlock(decoder_channels[0], decoder_channels[0])
        
        # 2. 上采样层 (与 TransCCV2 相同)
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(decoder_channels[0], decoder_channels[1])), # 32->64 (512->256)
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(decoder_channels[1], decoder_channels[2])), # 64->128 (256->128)
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(decoder_channels[2], decoder_channels[3])), # 128->256 (128->64)
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(decoder_channels[3], decoder_channels[3]))  # 256->512 (64->64)
        ])
        
        # 3. CBAM 块 (与 TransCCV2 相同)
        self.cbam_blocks = nn.ModuleList([
            CBAMResidualBlock(decoder_channels[0], decoder_channels[0]),
            CBAMResidualBlock(decoder_channels[1], decoder_channels[1]),
            CBAMResidualBlock(decoder_channels[2], decoder_channels[2])
        ])
        
        # 4. (修改点) 跳层连接，使用 CNNSkipConnection
        self.skip_connections = nn.ModuleList([
            # 融合 64x64 特征
            CNNSkipConnection(cnn_skip_channels[0], decoder_channels[1], decoder_channels[1]), # (c_cnn=192, c_dec=256, out=256)
            # 融合 128x128 特征
            CNNSkipConnection(cnn_skip_channels[1], decoder_channels[2], decoder_channels[2]), # (c_cnn=96, c_dec=128, out=128)
            # 融合 256x256 特征
            CNNSkipConnection(cnn_skip_channels[2], decoder_channels[3], decoder_channels[3])  # (c_cnn=48, c_dec=64, out=64)
        ])
        
        # 5. 特征聚合器 (与 TransCCV2 相同)
        self.feature_aggregators = nn.ModuleList([
            FeatureAggregator(decoder_channels[1], decoder_channels[1]),
            FeatureAggregator(decoder_channels[2], decoder_channels[2]),
            FeatureAggregator(decoder_channels[3], decoder_channels[3])
        ])
        
        # 6. 输出头 (与 TransCCV2 相同)
        self.seg_head_stage1 = nn.Sequential(ConvBlock(decoder_channels[1], decoder_channels[1] // 2), nn.Conv2d(decoder_channels[1] // 2, num_classes, 1))
        self.seg_head_stage2 = nn.Sequential(ConvBlock(decoder_channels[2], decoder_channels[2] // 2), nn.Conv2d(decoder_channels[2] // 2, num_classes, 1))
        self.final_proj = ASPP(decoder_channels[3], decoder_channels[3] // 2)
        self.seg_head_main = nn.Conv2d(decoder_channels[3] // 2, num_classes, 1)
        self.boundary_head_aux = nn.Sequential(ConvBlock(decoder_channels[2], decoder_channels[2] // 2), nn.Conv2d(decoder_channels[2] // 2, 1, 1))
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
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, transformer_features, cnn_skip_features):
        B, _, _ = transformer_features.shape
        # (B, N+1, C) -> (B, N, C)
        patch_features = transformer_features[:, 1:, :] 
        
        # (B, N, C) -> (B, N, C_dec[0])
        features = self.feature_proj(patch_features) 
        H = W = int(features.shape[1] ** 0.5) # 32x32
        
        # (B, N, C_dec[0]) -> (B, C_dec[0], H, W)
        x = features.transpose(1, 2).reshape(B, self.decoder_channels[0], H, W)
        x = self.initial_conv(x) # (B, 512, 32, 32)
        
        # --- Stage 1 ---
        x = self.cbam_blocks[0](x)
        x = self.upsample_layers[0](x) # (B, 256, 64, 64)
        # 融合 CNN skip[0] (64x64)
        x = self.skip_connections[0](cnn_skip_features[0], x) 
        x = self.feature_aggregators[0](x)
        seg_aux1 = self.seg_head_stage1(x)

        # --- Stage 2 ---
        x = self.cbam_blocks[1](x)
        x = self.upsample_layers[1](x) # (B, 128, 128, 128)
        # 融合 CNN skip[1] (128x128)
        x = self.skip_connections[1](cnn_skip_features[1], x)
        x = self.feature_aggregators[1](x)
        seg_aux2 = self.seg_head_stage2(x)

        # --- Stage 3 ---
        x = self.cbam_blocks[2](x)
        boundary_aux = self.boundary_head_aux(x)
        x = self.upsample_layers[2](x) # (B, 64, 256, 256)
        # 融合 CNN skip[2] (256x256)
        x = self.skip_connections[2](cnn_skip_features[2], x)
        x = self.feature_aggregators[2](x)

        # --- Final stage ---
        x = self.upsample_layers[3](x) # (B, 64, 512, 512)
        x = self.dropout(x)
        x = self.final_proj(x)
        seg_main = self.seg_head_main(x)
        boundary_main = self.boundary_head_main(x)

        # 上采样所有输出到原始尺寸
        seg_main_out = F.interpolate(seg_main, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        seg_aux1_out = F.interpolate(seg_aux1, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        seg_aux2_out = F.interpolate(seg_aux2, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        boundary_aux_out = F.interpolate(boundary_aux, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        boundary_main_out = F.interpolate(boundary_main, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return seg_main_out, boundary_main_out, seg_aux1_out, seg_aux2_out, boundary_aux_out


class TransCCV3(nn.Module):
    """
    (新) 混合模型
    - Transformer 编码器 (全局)
    - CNN 编码器 (局部)
    - 混合解码器 (融合)
    """
    def __init__(self, img_size: int = 512, patch_size: int = 16, in_chans: int = 3, num_classes: int = 2,
                 embed_dim: int = 768, depth: int = 6, num_heads: int = 12, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1, use_fourier_pos: bool = True,
                 decoder_channels: List[int] = [512, 256, 128, 64],
                 hdnet_base_channel: int = 48):
        super().__init__()
        
        # 1. 全局流: Transformer 编码器
        self.transformer_encoder = TransformerEncoder(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
            drop_rate, attn_drop_rate, drop_path_rate, use_fourier_pos=use_fourier_pos
        )
        
        # 2. 局主流: CNN 编码器 (来自 HDNet)
        self.cnn_encoder = CNNEncoder(base_channel=hdnet_base_channel)
        
        # CNN skip 特征的通道数: [c64, c128, c256]
        cnn_skip_channels = [hdnet_base_channel * 4, hdnet_base_channel * 2, hdnet_base_channel]
        
        # 3. 混合解码器
        self.decoder = HybridDecoder(
            embed_dim, img_size, num_classes, decoder_channels, cnn_skip_channels
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # 1. 提取全局特征
        # (B, N+1, C)
        transformer_features = self.transformer_encoder(x)
        
        # 2. 提取局部特征
        # List[(B, 192, 64, 64), (B, 96, 128, 128), (B, 48, 256, 256)]
        cnn_skip_features = self.cnn_encoder(x)
        
        # 3. 解码器融合
        return self.decoder(transformer_features, cnn_skip_features)


def create_transcc_v3(config: Optional[dict] = None) -> HybridFormer:
    """创建 HybridFormer 模型的工厂函数"""
    if config is None:
        config = {
            'img_size': 512, 'patch_size': 16, 'in_chans': 3, 'num_classes': 2, 'embed_dim': 768,
            'depth': 6, 'num_heads': 12, 'mlp_ratio': 4.0, 'qkv_bias': True, 'drop_rate': 0.0,
            'attn_drop_rate': 0.0, 'drop_path_rate': 0.1, 'decoder_channels': [512, 256, 128, 64],
            'use_fourier_pos': True, 'hdnet_base_channel': 48,
        }
    return HybridFormer(**config)