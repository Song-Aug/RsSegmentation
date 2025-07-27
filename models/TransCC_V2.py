# transcc_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# ---------- 基础组件 ----------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class MLP(nn.Module):
    def __init__(self, in_features, hidden=None, out=None, act=nn.GELU, drop=0.):
        super().__init__()
        out = out or in_features
        hidden = hidden or in_features
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = act()
        self.fc2 = nn.Linear(hidden, out)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=512, patch_size=8, in_chans=3, embed=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # B N C

# ---------- 位置编码 ----------
class FourierPositionEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10000):
        super().__init__()
        assert dim % 4 == 0
        freq_dim = dim // 4
        bands = torch.exp(torch.linspace(0, math.log(max_freq), freq_dim))
        self.register_buffer('bands', bands)
        self.proj = nn.Linear(dim, dim)
    def get_grid(self, H, W, device):
        y, x = torch.meshgrid(torch.linspace(0,1,H,device=device),
                              torch.linspace(0,1,W,device=device), indexing='ij')
        grid = torch.stack([x.flatten(), y.flatten()], -1)  # (N,2)
        return grid
    def forward(self, B, H, W, device):
        grid = self.get_grid(H, W, device)  # (N,2)
        x, y = grid[:,0:1], grid[:,1:2]
        x_sin, x_cos = torch.sin(x*self.bands), torch.cos(x*self.bands)
        y_sin, y_cos = torch.sin(y*self.bands), torch.cos(y*self.bands)
        emb = torch.cat([x_sin,x_cos,y_sin,y_cos],-1)  # (N,dim)
        return self.proj(emb).unsqueeze(0).expand(B,-1,-1)

# ---------- 编码器 ----------
class Encoder(nn.Module):
    def __init__(self, img_size=512, patch_size=8, in_chans=3, embed=768, depth=6,
                 heads=12, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0.05, use_fourier=True):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed)
        N = (img_size//patch_size)**2
        self.cls = nn.Parameter(torch.zeros(1,1,embed))
        self.pos_drop = nn.Dropout(drop)
        self.use_fourier = use_fourier
        if use_fourier:
            self.pos_embed = FourierPositionEmbedding(embed)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1,N+1,embed))
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed, heads, mlp_ratio, qkv_bias, drop, attn_drop, dpr[i])
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed)
        self.apply(self._init_weights)
    def _init_weights(self,m):
        if isinstance(nn.Linear, type(m)):
            trunc_normal_(m.weight,std=.02)
            if m.bias is not None: nn.init.constant_(m.bias,0)
    def forward(self,x):
        B,_,H,W = x.shape
        z = self.patch_embed(x)           # B N C
        cls = self.cls.expand(B,-1,-1)
        z = torch.cat([cls,z],1)          # B N+1 C
        if self.use_fourier:
            patch_h = H // self.patch_embed.proj.kernel_size[0]
            patch_w = W // self.patch_embed.proj.kernel_size[1]
            pos = self.pos_embed(B, patch_h, patch_w, x.device)  # 动态尺寸
            z[:,1:] += pos
        else:
            z += self.pos_embed
        z = self.pos_drop(z)
        feats=[]
        for i,blk in enumerate(self.blocks):
            z = blk(z)
            if i in [1,3,5]:
                feats.append(z[:,1:])     # 去掉cls
        return self.norm(z), feats

# ---------- 解码器 ----------
class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))
    def forward(self,x): return self.conv(self.up(x))

class Skip(nn.Module):
    def __init__(self, enc_c, dec_c, out_c):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(enc_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))
        self.dec_conv = nn.Conv2d(dec_c, out_c, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c*2, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))
    def forward(self, enc, dec):
        B,N,C = enc.shape
        H = int(math.sqrt(N))
        enc = enc.transpose(1,2).view(B,C,H,H)
        enc = F.interpolate(enc, size=dec.shape[-2:], mode='bilinear', align_corners=False)
        enc = self.enc_conv(enc)
        dec = self.dec_conv(dec)
        return self.fuse(torch.cat([enc,dec],1))

class Decoder(nn.Module):
    def __init__(self, embed=768, patch_size=8, img_size=512, num_classes=2, chs=[512,256,128,64]):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(nn.Linear(embed, chs[0]), nn.LayerNorm(chs[0]))
        self.up1 = Up(chs[0], chs[1])
        self.skip1 = Skip(embed, chs[1], chs[1])
        self.up2 = Up(chs[1], chs[2])
        self.skip2 = Skip(embed, chs[2], chs[2])
        self.up3 = Up(chs[2], chs[3])
        self.skip3 = Skip(embed, chs[3], chs[3])
        self.up4 = Up(chs[3], chs[3])
        self.head = nn.Sequential(
            nn.Conv2d(chs[3], chs[3]//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(chs[3]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(chs[3]//2, num_classes, 1))
    def forward(self, z, feats):
        B,_,C = z.shape
        H = self.patch_size * int(math.sqrt(z.shape[1]-1))
        x = z[:,1:].transpose(1,2).view(B,C,H//self.patch_size, H//self.patch_size)
        x = self.proj(x.permute(0,2,3,1)).permute(0,3,1,2)   # B C h w
        x = self.up1(x)
        x = self.skip1(feats[0], x)
        x = self.up2(x)
        x = self.skip2(feats[1], x)
        x = self.up3(x)
        x = self.skip3(feats[2], x)
        x = self.up4(x)
        return self.head(x)

# ---------- 完整模型 ----------
class TransCC_v2(nn.Module):
    def __init__(self, img_size=512, patch_size=8, in_chans=3, num_classes=2,
                 embed=768, depth=6, heads=12, mlp_ratio=4., drop_path=0.05):
        super().__init__()
        self.encoder = Encoder(img_size, patch_size, in_chans, embed, depth, heads,
                               mlp_ratio, drop_path=drop_path)
        self.decoder = Decoder(embed, patch_size, img_size, num_classes)
    def forward(self, x):
        z, feats = self.encoder(x)
        return self.decoder(z, feats)

# ---------- 工厂函数 ----------
def create_transcc_model(cfg=None):
    if cfg is None:
        cfg = dict(patch_size=8, num_classes=2, drop_path=0.05)
    return TransCC_v2(**cfg)

create_transcc_3bands = lambda n=2: create_transcc_model({'num_classes':n})
create_transcc_4bands = lambda n=2: create_transcc_model({'in_chans':4,'num_classes':n})