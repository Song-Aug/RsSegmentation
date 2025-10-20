import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer


class GCFM(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(GCFM, self).__init__()
        self.k = k
        self.num_channels = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels, self.num_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, self.num_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, self.num_channels, kernel_size=1)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                self.num_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # As per equation (2)
        query = self.conv1(x).view(x.size(0), self.num_channels, -1)
        key = self.conv2(x).view(x.size(0), self.num_channels, -1).permute(0, 2, 1)
        correlation_matrix = torch.bmm(query, key)
        correlation_matrix = F.softmax(correlation_matrix, dim=-1)

        # As per equation (3)
        x_up = F.interpolate(
            x, scale_factor=self.k, mode="bilinear", align_corners=True
        )
        value = self.conv3(x_up).view(x.size(0), self.num_channels, -1)

        fused_features = torch.bmm(correlation_matrix, value)
        fused_features = fused_features.view(
            x.size(0), self.num_channels, x_up.size(2), x_up.size(3)
        )

        # Residual connection and final convolution
        output = self.fusion_conv(fused_features) + x_up
        return output


class GCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # As per equation (4)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # As per equation (5)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, high_level_features, low_level_features):
        # high_level_features: X (from deeper layer)
        # low_level_features: T (from shallower layer)

        # Concatenate and compute attention map A
        combined = torch.cat([high_level_features, low_level_features], dim=1)
        attention_map = self.gate_conv(combined)

        # Gating and residual connection
        gated_features = high_level_features * attention_map + high_level_features

        # Final convolution
        output = self.out_conv(gated_features)
        return output


class GlobalStream(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=512):
        super(GlobalStream, self).__init__()
        self.in_channels = encoder_channels  # [128, 256, 512, 1024]

        self.conv_e4 = nn.Conv2d(self.in_channels[3], decoder_channels, 1)
        self.conv_e3 = nn.Conv2d(self.in_channels[2], decoder_channels, 1)
        self.conv_e2 = nn.Conv2d(self.in_channels[1], decoder_channels, 1)
        self.conv_e1 = nn.Conv2d(self.in_channels[0], decoder_channels, 1)

        self.gcf_k8 = GCFM(decoder_channels, decoder_channels, k=8)
        self.gcf_k4 = GCFM(decoder_channels, decoder_channels, k=4)
        self.gcf_k2 = GCFM(decoder_channels, decoder_channels, k=2)

        self.fusion_conv = nn.Conv2d(decoder_channels * 3, decoder_channels, 1)

    def forward(self, features):
        e1, e2, e3, e4 = features

        g1 = self.conv_e4(e4)

        # Fusion with skip connections
        g2 = self.gcf_k2(g1) + self.conv_e3(e3)
        g3 = self.gcf_k4(g1) + self.conv_e2(e2)
        g4 = self.gcf_k8(g1) + self.conv_e1(e1)

        # Upsample G2 and G3 to match G4's resolution
        g2_up = F.interpolate(
            g2, size=g4.shape[2:], mode="bilinear", align_corners=True
        )
        g3_up = F.interpolate(
            g3, size=g4.shape[2:], mode="bilinear", align_corners=True
        )

        # Concatenate and final conv
        g_prime = self.fusion_conv(torch.cat([g2_up, g3_up, g4], dim=1))
        return g_prime


class ShapeStream(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=128):
        super(ShapeStream, self).__init__()
        self.in_channels = encoder_channels  # [128, 256, 512, 1024]

        # Prepare S1, S2, S3, S4 as per paper
        self.conv_s1 = nn.Conv2d(self.in_channels[0], decoder_channels, 1)
        self.conv_s2 = nn.Conv2d(self.in_channels[1], 1, 1)
        self.conv_s3 = nn.Conv2d(self.in_channels[2], 1, 1)
        self.conv_s4 = nn.Conv2d(self.in_channels[3], 1, 1)

        # Three cascaded GCMs
        self.gcm1 = GCM(decoder_channels, decoder_channels)
        self.gcm2 = GCM(decoder_channels, decoder_channels)
        self.gcm3 = GCM(decoder_channels, decoder_channels)

        self.fusion_conv = nn.Conv2d(decoder_channels * 3, decoder_channels, 1)

    def forward(self, features):
        e1, e2, e3, e4 = features
        target_size = e1.shape[2:]

        s1 = self.conv_s1(e1)
        s2 = F.interpolate(
            self.conv_s2(e2), size=target_size, mode="bilinear", align_corners=True
        )
        s3 = F.interpolate(
            self.conv_s3(e3), size=target_size, mode="bilinear", align_corners=True
        )
        s4 = F.interpolate(
            self.conv_s4(e4), size=target_size, mode="bilinear", align_corners=True
        )

        gcm1_out = self.gcm1(s1, s2)
        gcm2_out = self.gcm2(gcm1_out, s3)
        gcm3_out = self.gcm3(gcm2_out, s4)

        s_prime = self.fusion_conv(torch.cat([gcm1_out, gcm2_out, gcm3_out], dim=1))
        return s_prime


class STDSNet(nn.Module):
    def __init__(self, num_classes, image_size, pretrained=True):
        super(STDSNet, self).__init__()

        # 1. Encoder (Swin-B)
        self.encoder = SwinTransformer(
            img_size=image_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            patch_size=4,
            ape=False,
            drop_path_rate=0.3,
            use_checkpoint=False,
        )
        if pretrained:
            url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu"
            )["model"]
            self.encoder.load_state_dict(checkpoint, strict=False)
            print("Swin-B encoder weights loaded.")

        encoder_channels = [128, 256, 512, 1024]

        # 2. Global Stream Decoder
        self.global_stream = GlobalStream(encoder_channels, decoder_channels=512)

        # 3. Shape Stream Decoder
        self.shape_stream = ShapeStream(encoder_channels, decoder_channels=128)

        # Classifier heads
        self.global_classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        self.shape_classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        x = self.encoder.patch_embed(x)
        absolute_pos_embed = getattr(self.encoder, "absolute_pos_embed", None)
        if absolute_pos_embed is not None:
            if absolute_pos_embed.dim() == x.dim():
                x = x + absolute_pos_embed
            elif absolute_pos_embed.dim() == 3 and x.dim() == 4:
                ape = absolute_pos_embed
                ape_h = int(ape.shape[1] ** 0.5)
                ape_w = ape_h
                ape = ape.view(ape.shape[0], ape_h, ape_w, ape.shape[2])
                x = x + ape
            elif absolute_pos_embed.dim() == 4 and x.dim() == 3:
                ape = absolute_pos_embed
                x = x + ape.view(ape.shape[0], -1, ape.shape[-1])
        pos_drop = getattr(self.encoder, "pos_drop", None)
        if pos_drop is not None:
            x = pos_drop(x)

        if x.dim() == 4:
            if x.shape[1] == x.shape[-1]:
                # timm channels-last output: (B, H, W, C)
                B, H, W, C = x.shape
            else:
                # timm channels-first output: (B, C, H, W)
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1).contiguous()
                C = x.shape[-1]
        elif x.dim() == 3:
            B, L, C = x.shape
            grid_size = getattr(self.encoder.patch_embed, "grid_size", None)
            if grid_size is not None:
                H, W = grid_size
            else:
                H = W = int(L**0.5)
            x = x.view(B, H, W, C)
        else:
            raise ValueError("Unexpected tensor shape after patch embedding")

        features = []
        for layer in self.encoder.layers:
            try:
                x = layer(x)
            except TypeError:
                x = layer(x, H, W)

            if isinstance(x, tuple):
                if len(x) == 3:
                    x, H, W = x
                elif len(x) == 2:
                    x, size = x
                    if isinstance(size, (tuple, list)) and len(size) == 2:
                        H, W = size
                else:
                    x = x[0]

            if x.dim() == 3:
                B, L, C = x.shape
                H = W = int(L**0.5)
                x = x.view(B, H, W, C)
            elif x.dim() == 4:
                B, H, W, C = x.shape
            else:
                raise ValueError("Unexpected tensor shape inside Swin encoder layer")

            feature = x.permute(0, 3, 1, 2).contiguous()
            features.append(feature)

        e1, e2, e3, e4 = features

        # Decoder forward pass
        global_stream_out = self.global_stream(features)
        shape_stream_out = self.shape_stream(features)

        # Final predictions
        global_pred = self.global_classifier(global_stream_out)
        shape_pred = self.shape_classifier(shape_stream_out)

        # Upsample to original image size
        global_pred = F.interpolate(
            global_pred, size=input_size, mode="bilinear", align_corners=True
        )
        shape_pred = F.interpolate(
            shape_pred, size=input_size, mode="bilinear", align_corners=True
        )

        return global_pred, shape_pred


if __name__ == "__main__":
    # 1. 定义模型参数
    num_classes = 2  # 例如：背景 + 建筑物
    input_size = 512  # 输入图像尺寸

    # 2. 实例化模型
    # pretrained=False 避免在测试时下载权重
    model = STDSNet(num_classes=num_classes, image_size=input_size, pretrained=False)
    model.eval()  # 设置为评估模式

    # 3. 创建一个伪输入张量 (dummy input)
    # 形状: (batch_size, channels, height, width)
    batch_size = 2
    input_channels = 3
    dummy_input = torch.randn(batch_size, input_channels, input_size, input_size)

    print(f"模型: STDSNet")
    print(f"输入张量形状: {dummy_input.shape}")

    # 4. 执行前向传播
    with torch.no_grad():  # 在评估时不需要计算梯度
        global_pred, shape_pred = model(dummy_input)

    # 5. 打印输出形状
    print(f"全局流输出形状: {global_pred.shape}")
    print(f"形状流输出形状: {shape_pred.shape}")

    # 验证输出尺寸是否与输入匹配
    assert global_pred.shape == (batch_size, num_classes, input_size, input_size)
    assert shape_pred.shape == (batch_size, num_classes, input_size, input_size)
    print("\n前向传播成功，输出尺寸正确！")
