import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # 移除严格的尺寸检查，支持灵活的输入尺寸
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        return x


class FourierPositionEmbedding(nn.Module):
    def __init__(self, embed_dim, max_freq=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq

        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D Fourier embedding"
        
        self.freq_dim = embed_dim // 4  # 每个坐标轴使用 embed_dim/4 个频率
        
        freq_bands = torch.exp(torch.linspace(0, math.log(max_freq), self.freq_dim))
        self.register_buffer('freq_bands', freq_bands)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, patch_centers):
        """
        Args:
            patch_centers: (B, N, 2) - 归一化的 patch 中心坐标 [x, y]
        Returns:
            pos_embed: (B, N, embed_dim) - Fourier position embeddings
        """
        B, N, _ = patch_centers.shape
        
        # 分离 x 和 y 坐标
        x_coords = patch_centers[:, :, 0:1]  # (B, N, 1)
        y_coords = patch_centers[:, :, 1:2]  # (B, N, 1)
        
        # 为每个坐标生成 Fourier 特征
        x_freqs = x_coords * self.freq_bands.unsqueeze(0).unsqueeze(0)  # (B, N, freq_dim)
        y_freqs = y_coords * self.freq_bands.unsqueeze(0).unsqueeze(0)  # (B, N, freq_dim)
        
        # 计算正弦和余弦嵌入
        x_sin = torch.sin(x_freqs)  # (B, N, freq_dim)
        x_cos = torch.cos(x_freqs)  # (B, N, freq_dim)
        y_sin = torch.sin(y_freqs)  # (B, N, freq_dim)
        y_cos = torch.cos(y_freqs)  # (B, N, freq_dim)
        
        # 拼接所有 Fourier 特征
        fourier_features = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)  # (B, N, embed_dim)
        
        # 通过线性层投影到目标维度
        pos_embed = self.proj(fourier_features)
        
        return pos_embed
    
    def get_patch_centers(self, img_size, patch_size):
        """计算 patch 中心的归一化坐标"""
        img_h, img_w = to_2tuple(img_size)
        patch_h, patch_w = to_2tuple(patch_size)
        
        # 计算每个维度的 patch 数量
        num_patches_h = img_h // patch_h
        num_patches_w = img_w // patch_w
        
        # 创建网格坐标
        y_coords = torch.arange(num_patches_h, dtype=torch.float32)
        x_coords = torch.arange(num_patches_w, dtype=torch.float32)
        
        # 计算 patch 中心坐标（像素坐标）
        y_centers = (y_coords + 0.5) * patch_h
        x_centers = (x_coords + 0.5) * patch_w
        
        # 归一化到 [0, 1] 范围
        y_centers = y_centers / img_h
        x_centers = x_centers / img_w
        
        # 创建网格并展平
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing='ij')
        patch_centers = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (N, 2)
        
        return patch_centers


class Encoder(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768, depth=6,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_fourier_pos=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_fourier_pos = use_fourier_pos
        
        # 图像块嵌入
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        if use_fourier_pos:
            self.pos_embed = FourierPositionEmbedding(embed_dim)
            # 为 CLS token 创建可学习的位置编码
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            # 传统的可学习位置编码
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer 块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # 最终归一化层
        self.norm = norm_layer(embed_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 图像块嵌入
        x = self.patch_embed(x)  # B, N, C
        
        # 添加类别令牌
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # B, N+1, C
        
        # 添加位置编码
        if self.use_fourier_pos:
            # 获取 patch 中心坐标
            patch_centers = self.pos_embed.get_patch_centers(
                img_size=(H, W), patch_size=self.patch_size
            ).to(x.device)
            
            # 扩展到 batch 维度
            patch_centers = patch_centers.unsqueeze(0).expand(B, -1, -1)
            
            # 计算 Fourier 位置编码
            fourier_pos = self.pos_embed(patch_centers)  # B, N, C
            
            # 为 patch tokens 添加 Fourier 位置编码
            x[:, 1:, :] = x[:, 1:, :] + fourier_pos
            
            # 为 CLS token 添加可学习的位置编码
            x[:, 0:1, :] = x[:, 0:1, :] + self.cls_pos_embed
        else:
            # 传统位置编码
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
        # 存储中间层特征用于跳级连接
        layer_features = []
        
        # 通过 Transformer 块，收集中间层特征
        for i, block in enumerate(self.blocks):
            x = block(x)
            # 收集特定层的特征用于跳级连接
            if i in [1, 3, 5]:  # 在第2, 4, 6层收集特征
                layer_features.append(x[:, 1:, :])  # 只保存patch features，去除CLS token
        
        # 最终归一化
        x = self.norm(x)
        
        return x, layer_features
    
    def get_patch_features(self, x):
        """获取不包含 CLS token 的补丁特征"""
        features, _ = self.forward(x)
        return features[:, 1:, :]  # 移除 CLS token
    
    def get_cls_token(self, x):
        """获取 CLS token"""
        features, _ = self.forward(x)
        return features[:, 0, :]  # 只返回 CLS token


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


class ConvBlock(nn.Module):
    """基础卷积块：Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CBAMResidualBlock(nn.Module):
    """带有 CBAM 的残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(CBAMResidualBlock, self).__init__()
        
        # 主分支
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # CBAM 注意力
        self.cbam = CBAM(out_channels)
        
        # 跳跃连接
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
        """
        Args:
            encoder_feat: (B, N, encoder_channels) - 编码器特征
            decoder_feat: (B, decoder_channels, H, W) - 解码器特征
        Returns:
            fused_feat: (B, out_channels, H, W) - 融合后的特征
        """
        B, N, encoder_channels = encoder_feat.shape
        B, decoder_channels, H, W = decoder_feat.shape
        
        # 将编码器特征投影并重塑为2D
        encoder_proj = self.encoder_proj(encoder_feat)  # (B, N, out_channels)
        encoder_2d = encoder_proj.transpose(1, 2).reshape(B, -1, int(N**0.5), int(N**0.5))  # (B, out_channels, h, w)
        
        # 上采样编码器特征到解码器特征的尺寸
        encoder_upsampled = F.interpolate(encoder_2d, size=(H, W), mode='bilinear', align_corners=False)
        
        # 投影解码器特征
        decoder_proj = self.decoder_conv(decoder_feat)  # (B, out_channels, H, W)
        
        # 特征融合
        fused = torch.cat([encoder_upsampled, decoder_proj], dim=1)  # (B, out_channels*2, H, W)
        fused = self.fusion_conv(fused)  # (B, out_channels, H, W)
        
        # 应用注意力机制
        fused = self.attention(fused)
        
        return fused


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


class Decoder(nn.Module):
    """TransCC 解码器：CNN + CBAM + 跳级连接"""
    
    def __init__(self, embed_dim=768, patch_size=16, img_size=512, num_classes=2, 
                 decoder_channels=[512, 256, 128, 64]):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels
        
        # 计算特征图尺寸
        self.feature_size = img_size // patch_size  # 512 // 16 = 32
        
        # 将 Transformer 特征重塑为 2D 特征图的投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, decoder_channels[0]),
            nn.LayerNorm(decoder_channels[0])
        )
        
        # 初始卷积层
        self.initial_conv = ConvBlock(decoder_channels[0], decoder_channels[0])
        
        # 3个 CBAM 残差块堆叠
        self.cbam_blocks = nn.ModuleList([
            CBAMResidualBlock(decoder_channels[0], decoder_channels[0]),  # 512 -> 512
            CBAMResidualBlock(decoder_channels[1], decoder_channels[1]),  # 256 -> 256 (上采样后的通道数)
            CBAMResidualBlock(decoder_channels[2], decoder_channels[2])   # 128 -> 128 (上采样后的通道数)
        ])
        
        # 上采样层（反卷积）
        self.upsample_layers = nn.ModuleList([
            # 从 32x32 -> 64x64
            nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], 
                             kernel_size=4, stride=2, padding=1, bias=False),
            # 从 64x64 -> 128x128
            nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], 
                             kernel_size=4, stride=2, padding=1, bias=False),
            # 从 128x128 -> 256x256
            nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], 
                             kernel_size=4, stride=2, padding=1, bias=False),
            # 从 256x256 -> 512x512
            nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3], 
                             kernel_size=4, stride=2, padding=1, bias=False)
        ])
        
        # 批归一化层
        self.upsample_bn = nn.ModuleList([
            nn.BatchNorm2d(decoder_channels[1]),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.BatchNorm2d(decoder_channels[3])
        ])
        
        # 跳级连接模块 - 融合编码器多层特征
        self.skip_connections = nn.ModuleList([
            # 第1个跳级连接: 64x64 分辨率，融合第2层编码器特征
            SkipConnection(embed_dim, decoder_channels[1], decoder_channels[1]),
            # 第2个跳级连接: 128x128 分辨率，融合第4层编码器特征  
            SkipConnection(embed_dim, decoder_channels[2], decoder_channels[2]),
            # 第3个跳级连接: 256x256 分辨率，融合第6层编码器特征
            SkipConnection(embed_dim, decoder_channels[3], decoder_channels[3])
        ])
        
        # 特征聚合模块
        self.feature_aggregators = nn.ModuleList([
            FeatureAggregator(decoder_channels[1], decoder_channels[1]),
            FeatureAggregator(decoder_channels[2], decoder_channels[2]), 
            FeatureAggregator(decoder_channels[3], decoder_channels[3])
        ])
        
        # 最终分类头
        self.final_conv = nn.Sequential(
            ConvBlock(decoder_channels[3], decoder_channels[3] // 2),
            ConvBlock(decoder_channels[3] // 2, decoder_channels[3] // 4),
            nn.Conv2d(decoder_channels[3] // 4, num_classes, kernel_size=1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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
    
    def forward(self, encoder_features, layer_features):
        """
        Args:
            encoder_features: (B, N+1, C) - 来自编码器的最终特征，包含 CLS token
            layer_features: List[(B, N, C)] - 编码器中间层特征，用于跳级连接
        Returns:
            output: (B, num_classes, H, W) - 分割结果
        """
        B, N_plus_1, C = encoder_features.shape
        
        # 移除 CLS token，只保留 patch features
        patch_features = encoder_features[:, 1:, :]  # (B, N, C)
        N = patch_features.shape[1]
        
        # 投影到解码器维度
        features = self.feature_proj(patch_features)  # (B, N, decoder_channels[0])
        
        # 重塑为 2D 特征图
        H = W = int(N ** 0.5)  # 假设是正方形特征图
        features = features.transpose(1, 2).reshape(B, self.decoder_channels[0], H, W)
        
        # 初始卷积
        x = self.initial_conv(features)  # (B, 512, 32, 32)
        
        # ====== 解码器主体 + 跳级连接 ======
        
        # 第一个 CBAM 块 + 上采样 (32x32 -> 64x64)
        x = self.cbam_blocks[0](x)  # (B, 512, 32, 32)
        x = F.relu(self.upsample_bn[0](self.upsample_layers[0](x)))  # (B, 256, 64, 64)
        
        # 跳级连接1: 融合第2层编码器特征
        if len(layer_features) >= 1:
            x = self.skip_connections[0](layer_features[0], x)  # 融合编码器第2层特征
        x = self.feature_aggregators[0](x)  # 特征聚合
        
        # 第二个 CBAM 块 + 上采样 (64x64 -> 128x128)
        x = self.cbam_blocks[1](x)  # (B, 256, 64, 64)
        x = F.relu(self.upsample_bn[1](self.upsample_layers[1](x)))  # (B, 128, 128, 128)
        
        # 跳级连接2: 融合第4层编码器特征
        if len(layer_features) >= 2:
            x = self.skip_connections[1](layer_features[1], x)  # 融合编码器第4层特征
        x = self.feature_aggregators[1](x)  # 特征聚合
        
        # 第三个 CBAM 块 + 上采样 (128x128 -> 256x256)
        x = self.cbam_blocks[2](x)  # (B, 128, 128, 128)
        x = F.relu(self.upsample_bn[2](self.upsample_layers[2](x)))  # (B, 64, 256, 256)
        
        # 跳级连接3: 融合第6层编码器特征
        if len(layer_features) >= 3:
            x = self.skip_connections[2](layer_features[2], x)  # 融合编码器第6层特征
        x = self.feature_aggregators[2](x)  # 特征聚合
        
        # 最终上采样到原图尺寸 (256x256 -> 512x512)
        x = F.relu(self.upsample_bn[3](self.upsample_layers[3](x)))  # (B, 64, 512, 512)
        
        # 最终分类
        output = self.final_conv(x)  # (B, num_classes, 512, 512)
        
        return output


class TransCC(nn.Module):
    """完整的 TransCC 模型：Transformer Encoder + CNN Decoder with CBAM"""
    
    def __init__(self, img_size=512, patch_size=16, in_chans=3, num_classes=2, 
                 embed_dim=768, depth=6, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 decoder_channels=[512, 256, 128, 64], use_fourier_pos=True):
        """
        Args:
            img_size (int): 输入图像尺寸
            patch_size (int): 图像块大小
            in_chans (int): 输入通道数 (3 for RGB, 4 for RGB+NIR)
            num_classes (int): 分类类别数
            embed_dim (int): 嵌入维度
            depth (int): Transformer 层数
            num_heads (int): 注意力头数
            mlp_ratio (float): MLP 隐藏层比例
            qkv_bias (bool): 是否使用 QKV bias
            drop_rate (float): Dropout 比例
            attn_drop_rate (float): 注意力 Dropout 比例
            drop_path_rate (float): Drop Path 比例
            decoder_channels (list): 解码器通道数配置
            use_fourier_pos (bool): 是否使用 Fourier 位置编码
        """
        super(TransCC, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Transformer 编码器
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
            use_fourier_pos=use_fourier_pos
        )
        
        # CNN 解码器
        self.decoder = Decoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            num_classes=num_classes,
            decoder_channels=decoder_channels
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - 输入图像
        Returns:
            output: (B, num_classes, H, W) - 分割结果
        """
        # 编码器前向传播，获取最终特征和中间层特征
        encoder_features, layer_features = self.encoder(x)  # (B, N+1, embed_dim), List[(B, N, embed_dim)]
        
        # 解码器前向传播，使用跳级连接
        output = self.decoder(encoder_features, layer_features)  # (B, num_classes, H, W)
        
        return output
    
    def get_encoder_features(self, x):
        """获取编码器特征，用于分析或可视化"""
        encoder_features, layer_features = self.encoder(x)
        return encoder_features, layer_features
    
    def get_patch_features(self, x):
        """获取 patch 特征（不包含 CLS token）"""
        return self.encoder.get_patch_features(x)
    
    def get_cls_token(self, x):
        """获取 CLS token 特征"""
        return self.encoder.get_cls_token(x)


def create_transcc_model(config=None):
    """
    创建 TransCC 模型的工厂函数
    
    Args:
        config (dict): 模型配置字典
    Returns:
        model: TransCC 模型实例
    """
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
            'use_fourier_pos': True
        }
    
    model = TransCC(**config)
    return model


def create_transcc_3bands(num_classes=2):
    """创建 3 通道（RGB）TransCC 模型"""
    config = {
        'img_size': 512,
        'patch_size': 16,
        'in_chans': 3,
        'num_classes': num_classes,
        'embed_dim': 768,
        'depth': 6,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'decoder_channels': [512, 256, 128, 64],
        'use_fourier_pos': True
    }
    return TransCC(**config)


def create_transcc_4bands(num_classes=2):
    """创建 4 通道（RGB+NIR）TransCC 模型"""
    config = {
        'img_size': 512,
        'patch_size': 16,
        'in_chans': 4,
        'num_classes': num_classes,
        'embed_dim': 768,
        'depth': 6,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'decoder_channels': [512, 256, 128, 64],
        'use_fourier_pos': True
    }
    return TransCC(**config)


