
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 卷积层，带填充"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 卷积层"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True, recompute_scale_factor=True)
        return x


# Index Pooling Module（索引池化模块）
class pool(nn.Module):
    def __init__(self, channels):
        super(pool, self).__init__()
        self.channels = channels
        # 在适当的设备上初始化权重
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight1 = torch.zeros((channels, 1, 2, 2), device=device)
        self.weight2 = torch.zeros((channels, 1, 2, 2), device=device)
        self.weight3 = torch.zeros((channels, 1, 2, 2), device=device)
        self.weight4 = torch.zeros((channels, 1, 2, 2), device=device)
        self.weight1[:, :, 0, 0] = 1
        self.weight2[:, :, 0, 1] = 1
        self.weight3[:, :, 1, 0] = 1
        self.weight4[:, :, 1, 1] = 1

    def forward(self, x):
        device = x.device
        # 将权重移动到与输入相同的设备
        self.weight1 = self.weight1.to(device)
        self.weight2 = self.weight2.to(device)
        self.weight3 = self.weight3.to(device)
        self.weight4 = self.weight4.to(device)
        
        with torch.no_grad():
            x1 = F.conv2d(x, self.weight1, stride=2, groups=self.channels, bias=None)
            x2 = F.conv2d(x, self.weight2, stride=2, groups=self.channels, bias=None)
            x3 = F.conv2d(x, self.weight3, stride=2, groups=self.channels, bias=None)
            x4 = F.conv2d(x, self.weight4, stride=2, groups=self.channels, bias=None)
        return x1, x2, x3, x4


# DAMIP Module (Dilated Attention Multi-Index Pooling)（空洞注意力多索引池化模块）
class attn_pool(nn.Module):
    def __init__(self, feature_channels):
        super(attn_pool, self).__init__()
        self.pool1 = pool(feature_channels)
        self.pool2 = pool(1)
        self.conv1 = nn.Conv2d(feature_channels * 4, 2 * feature_channels, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(2 * feature_channels, 2 * feature_channels, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(2 * feature_channels)
        self.a1 = nn.Parameter(torch.Tensor(1))
        self.a2 = nn.Parameter(torch.Tensor(1))
        self.a3 = nn.Parameter(torch.Tensor(1))
        self.a4 = nn.Parameter(torch.Tensor(1))

    def forward(self, map, feature):
        feature1, feature2, feature3, feature4 = self.pool1(feature)
        map1, map2, map3, map4 = self.pool2(map)

        fm1 = self.a1 * feature1 + feature1 * map1
        fm2 = self.a2 * feature2 + feature2 * map2
        fm3 = self.a3 * feature3 + feature3 * map3
        fm4 = self.a4 * feature4 + feature4 * map4

        mat = torch.cat((fm1, fm2, fm3, fm4), 1)
        mat = self.conv1(mat)
        mat = F.relu(self.bn2(self.conv2(mat)))
        return mat


# DPMG Module (Deep Pyramid Multi-path Guidance)（深度金字塔多路径引导模块）
class dsup(nn.Module):
    def __init__(self, input_channels):
        super(dsup, self).__init__()
        self.conv1 = conv3x3(input_channels, input_channels // 2)
        self.bn1 = nn.BatchNorm2d(input_channels // 2)
        self.conv2 = conv3x3(input_channels // 2, 32)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.sigmoid(x)


# Dilated Convolutional Block（空洞卷积块）
class conv_enc(nn.Module):
    def __init__(self, in_channels, out_channels, dil):
        super(conv_enc, self).__init__()
        self.conv1 = conv1x1(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = conv3x3(in_channels, in_channels, dilation=dil)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = conv1x1(in_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = conv3x3(out_channels, out_channels, dilation=dil)
        self.bn4 = nn.BatchNorm2d(out_channels)
        # 输入维度匹配
        self.conv0 = conv1x1(in_channels, out_channels)

    def forward(self, x):
        identity = self.conv0(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        return F.relu(x + identity)


# Feature Extractor（特征提取器）
class enc(nn.Module):
    def __init__(self, input_channels, output_channels, dil):
        super(enc, self).__init__()
        self.conv1 = conv_enc(input_channels, output_channels, dil)
        self.conv2 = conv_enc(output_channels, output_channels, 2 * dil)
        self.dp_sup = dsup(output_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1_out = self.dp_sup(x1)
        return x1, x1_out


# DAMSCA Module (Dilated Attention Multi-Scale Channel Attention)（空洞注意力多尺度通道注意力模块）
class kqcbam(nn.Module):
    def __init__(self, input_channels, scale_factor=2):
        super(kqcbam, self).__init__()
        self.conv1 = nn.Conv2d(1, input_channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.upsample = Upsample(scale_factor)

    def forward(self, map, feature):
        f1 = map * feature
        map2 = self.conv1(map)
        map2 = self.gap(map2)
        f2 = torch.sigmoid(map2) * feature
        out = F.relu(f1 + f2)
        return self.upsample(out)


# Decoder Module（解码器模块）
class decoder(nn.Module):
    def __init__(self, input_channels):
        super(decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = self.conv_out(x)
        return torch.sigmoid(x)


# MSSDMPA-Net Main Architecture（MSSDMPA-Net 主架构）
class MSSDMPA_Net(nn.Module):
    """
    MSSDMPA-Net: Multi-Scale Spatial-Dependent Multi-Path Attention Network
    多尺度空间相关多路径注意力网络
    
    参数：
        input_channels (int): 输入通道数（默认：3，用于RGB图像）
        num_classes (int): 输出类别数（默认：1，用于二分类分割）
    """
    def __init__(self, input_channels=3, num_classes=1):
        super(MSSDMPA_Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.pool1 = attn_pool(64)
        self.pool2 = attn_pool(128)
        self.pool3 = attn_pool(256)

        self.path1 = enc(64, 64, 1)
        self.path2 = enc(128, 128, 2)
        self.path3 = enc(256, 256, 3)
        self.path4 = enc(512, 512, 4)

        self.cbm1 = kqcbam(64, 1)
        self.cbm2 = kqcbam(128, 2)
        self.cbm3 = kqcbam(256, 4)
        self.cbm4 = kqcbam(512, 8)

        self.decoder = decoder(960)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x1, x1_out = self.path1(x)
        x = self.pool1(x1_out, x)
        x2, x2_out = self.path2(x)
        x = self.pool2(x2_out, x)
        x3, x3_out = self.path3(x)
        x = self.pool3(x3_out, x)
        x4, x4_out = self.path4(x)

        x1 = self.cbm1(x1_out, x1)
        x2 = self.cbm2(x2_out, x2)
        x3 = self.cbm3(x3_out, x3)
        x4 = self.cbm4(x4_out, x4)
        x_out = torch.cat((x1, x2, x3, x4), 1)
        x_out = self.decoder(x_out)
        return x_out, x1_out, x2_out, x3_out, x4_out


class dsmpnet(MSSDMPA_Net):
    """MSSDMPA_Net 的别名，用于保持与原始代码的兼容性"""
    pass


# Loss Functions（损失函数）
class gen_loss(nn.Module):
    def __init__(self, gamma=1.5, batch=True):
        super(gen_loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.gamma = gamma

    def gen_dice(self, y_pred, y_true):
        epsilon = 1e-8
        l1 = abs(y_pred - y_true) ** self.gamma
        y_pred_sqsum = torch.sum((y_pred * y_pred))
        y_true_sqsum = torch.sum((y_true * y_true))
        l1_sum = torch.sum(l1)
        score = (l1_sum + epsilon) / (y_pred_sqsum + y_true_sqsum)
        return score.mean()

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.gen_dice(y_pred, y_true)
        return a + b


def y_bce_loss(prediction1, prediction2, prediction3, prediction4, prediction5, label):
    """MSSDMPA-Net 的多尺度损失函数"""
    dice = gen_loss()
    loss1 = dice(prediction1, label)
    label = torch.nn.functional.interpolate(label, size=(256, 256), scale_factor=None, mode='nearest')
    loss2 = dice(prediction2, label)
    label = torch.nn.functional.interpolate(label, size=(128, 128), scale_factor=None, mode='nearest')
    loss3 = dice(prediction3, label)
    label = torch.nn.functional.interpolate(label, size=(64, 64), scale_factor=None, mode='nearest')
    loss4 = dice(prediction4, label)
    label = torch.nn.functional.interpolate(label, size=(32, 32), scale_factor=None, mode='nearest')
    loss5 = dice(prediction5, label)
    loss = loss1 + loss2 + loss3 + loss4 + loss5
    return loss


class IoU(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoU, self).__init__()
        self.threshold = threshold

    def forward(self, target, input):
        eps = 1e-10
        input_ = (input > self.threshold).data.float()
        target_ = (target > self.threshold).data.float()

        intersection = torch.clamp(input_ * target_, 0, 1)
        union = torch.clamp(input_ + target_, 0, 1)

        if torch.mean(intersection).lt(eps):
            return torch.Tensor([0., 0., 0., 0.])
        else:
            acc = torch.mean((input_ == target_).data.float())
            iou = torch.mean(intersection) / torch.mean(union)
            recall = torch.mean(intersection) / torch.mean(target_)
            precision = torch.mean(intersection) / torch.mean(input_)
            return torch.Tensor([acc, recall, precision, iou])


def dice_coeff(y_true, y_pred, batch=True):
    """Dice 系数计算"""
    smooth = 1e-8
    if batch:
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
    else:
        i = y_true.sum(1).sum(1).sum(1)
        j = y_pred.sum(1).sum(1).sum(1)
        intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
    score = (2. * intersection + smooth) / (i + j + smooth)
    return score.mean()


# Training Functions（训练函数）
def train_one_epoch_net(model, train_dl, learn):
    """训练模型一个epoch"""
    opt = torch.optim.Adam(model.parameters(), lr=learn)
    running_loss_image = 0.0
    metric_epoch = 0.0
    dice_epoch = 0.0
    iou_metric = IoU()
    
    model.train()
    for a, b in train_dl:
        a = a.float()
        label = b.float()
        label_loss = b.type(torch.float)
        
        device = next(model.parameters()).device
        a = a.to(device)
        label_loss = label_loss.to(device)
        
        pred1, pred2, pred3, pred4, pred5 = model(a)
        loss = y_bce_loss(pred1, pred2, pred3, pred4, pred5, label_loss)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        running_loss_image += loss.item()
        metric = iou_metric(label_loss.cpu(), pred1.detach().cpu())
        dice = dice_coeff(label_loss.cpu(), pred1.detach().cpu())
        dice_epoch += dice
        metric_epoch += metric
    
    running_loss_image /= len(train_dl)
    metric_epoch /= len(train_dl)
    dice_epoch /= len(train_dl)
    return model, dice_epoch, metric_epoch, running_loss_image


def validate_one_epoch_net(model, val_dl):
    """验证模型一个epoch"""
    running_loss_image = 0.0
    metric_epoch = 0.0
    dice_epoch = 0.0
    iou_metric = IoU()
    
    model.eval()
    with torch.no_grad():
        for a, b in val_dl:
            a = a.float()
            label = b.float()
            label_loss = b.type(torch.float)
            
            device = next(model.parameters()).device
            a = a.to(device)
            label_loss = label_loss.to(device)
            
            pred1, pred2, pred3, pred4, pred5 = model(a)
            loss = y_bce_loss(pred1, pred2, pred3, pred4, pred5, label_loss)
            
            running_loss_image += loss.item()
            metric = iou_metric(label_loss.cpu(), pred1.detach().cpu())
            dice = dice_coeff(label_loss.cpu(), pred1.detach().cpu())
            dice_epoch += dice
            metric_epoch += metric
    
    running_loss_image /= len(val_dl)
    metric_epoch /= len(val_dl)
    dice_epoch /= len(val_dl)
    return dice_epoch, metric_epoch, running_loss_image


def train_epoches_net(model, train_dl, test_dl, epochs, learn_rate, save_path):
    """训练模型多个epoch"""
    max_accuracy = 0.0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for i in range(epochs):
        model, dice_train, iou_train, loss_train = train_one_epoch_net(model, train_dl, learn_rate)
        dice_test, iou_test, loss_test = validate_one_epoch_net(model, test_dl)
        
        print('第 ' + str(i + 1) + ' 个epoch完成')
        print(f'train_loss: {loss_train:.6f}, train_dice: {dice_train:.6f}, train_iou: {iou_train[3]:.6f}')
        print(f'test_loss: {loss_test:.6f}, test_dice: {dice_test:.6f}, test_iou: {iou_test[3]:.6f}')
        
        path_final = os.path.join(save_path, f"epoch{i}_test_loss{loss_test:.4f}.pth")
        torch.save(model.state_dict(), path_final)
        
        if iou_test[3] > max_accuracy:
            max_accuracy = iou_test[3]
            best_path = os.path.join(save_path, f"best_epoch{i}_test_iou{iou_test[3]:.4f}.pth")
            torch.save(model.state_dict(), best_path)


def create_model(input_channels=3, num_classes=1, device='cuda'):
    """创建并初始化 MSSDMPA-Net 模型"""
    model = MSSDMPA_Net(input_channels=input_channels, num_classes=num_classes)
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    elif device == 'cpu':
        model = model.cpu()
    return model


# Example usage（使用示例）
if __name__ == "__main__":
    # 创建模型
    model = create_model(input_channels=3, num_classes=1)
    
    # 打印模型信息
    print("MSSDMPA-Net 模型创建成功!")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 使用随机输入测试
    if torch.cuda.is_available():
        test_input = torch.randn(1, 3, 512, 512).cuda()
        with torch.no_grad():
            output = model(test_input)
            print(f"输入形状: {test_input.shape}")
            print(f"输出形状: {[o.shape for o in output]}")
    else:
        print("CUDA 不可用，使用 CPU 创建模型。")