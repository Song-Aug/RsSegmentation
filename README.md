# BuildingSeg Project

这是一个基于 PyTorch 的遥感图像分割项目，主要针对建筑物提取任务（Building Segmentation）。项目使用了 TransCC V3 模型架构，并集成了 WandB 实验追踪和飞书（Lark）通知功能。

## 1. 依赖库 (Dependencies)

本项目主要依赖以下 Python 库：

- **深度学习框架**:
  - `torch` >= 1.x
  - `torchvision`
- **数据处理与增强**:
  - `numpy`
  - `Pillow`
  - `rasterio` (用于读取多光谱 TIFF 图像)
  - `albumentations` (用于图像数据增强)
- **可视化与进度**:
  - `matplotlib`
  - `tqdm`
- **实验记录与通知**:
  - `wandb` (Weights & Biases, 用于实验监控)
  - `requests` (用于飞书通知)

安装示例:
```bash
pip install torch torchvision numpy Pillow rasterio albumentations matplotlib tqdm wandb requests
```

## 2. 输入输出说明 (Input & Output)

### 输入数据 (Input)
项目通过配置文件 `configs/transcc_v3_config.py` 中的 `data_root` 参数指定数据根目录。
默认路径结构如下：

```text
data_root/
├── Train/
│   ├── image/  # 存放训练集图像 (.tif, .tiff)
│   └── label/  # 存放训练集标签 (单通道灰度图)
├── Val/
│   ├── image/
│   └── label/
├── Test/
│   ├── image/
│   └── label/
└── Vis/        # (可选) 用于可视化测试的样本
    ├── image/
    └── label/
```

- **图像格式**: 支持多波段 TIFF 图像 (3波段 RGB 或 4波段 RGB+NIR)。
  - 若 `config['use_nir'] = True`，则读取前4个通道。
  - 若 `config['use_nir'] = False`，则只读取前3个通道 (RGB)。
- **标签格式**: 单通道灰度图像，像素值为类别索引 (0为背景，1为建筑物)。文件名称必须与对应的图像文件一致。

### 输出文件 (Output)

- **模型权重**: 保存于 `./checkpoints/<Experiment_Name>/` 目录下。
  - `best_model.pth`: 验证集 IoU 最高的模型权重。
- **日志文件**:
  - 本地日志: `./checkpoints/<Experiment_Name>/train_log.txt`。
  - 云端日志: 同步至 WandB 项目 `Building-Segmentation-3Bands`。
- **可视化结果**:
  - 训练过程中每 10 个 Epoch 会在 WandB 上生成并记录预测结果对比图。

## 3. 运行指南 (Usage)

### 配置
在 `configs/transcc_v3_config.py` 中修改训练参数，例如：
- `data_root`: 数据集路径
- `batch_size`: 批次大小
- `num_epochs`: 训练总轮数
- `learning_rate`: 学习率
- `model_name`: 实验名称标识

### 训练
运行训练脚本：
```bash
python train/train_transccV3.py
```
脚本会自动初始化 WandB，加载数据，开始训练，并在满足条件时发送飞书通知。

## 4. 其他特性 (Features)

- **混合精度训练**: 默认启用 `torch.cuda.amp` 进行自动混合精度训练，节省显存并加速。
- **两阶段增强**: 
  - 前期 (`mild_aug_epoch` 之前) 使用强数据增强 (Strong Augmentation)。
  - 后期使用温和数据增强 (Mild Augmentation) 以稳定模型收敛。
- **飞书通知**: 训练开始、发现更优模型、训练结束或报错时，会自动发送飞书消息通知 (需在 `utils4train/alerts_by_lark.py` 中配置 Webhook)。

## 5. 推理 (Inference)

项目提供了大图滑窗推理脚本 `infer/infer_transccv3.py`，支持地理参考信息的保持。

### 使用方法
1. 修改 `infer/infer_transccv3.py` 中的文件路径配置：
   ```python
   # --- 1. 参数设置 ---
   model_path = "/path/to/your/checkpoint.pth"  # 模型权重路径
   input_path = "/path/to/your/image.tif"        # 待预测大图路径
   output_path = "/path/to/save/result.tif"      # 结果保存路径
   ```
2. 运行推理脚本：
   ```bash
   python infer/infer_transccv3.py
   ```

### 特性
- **滑窗推理**: 针对大尺寸遥感影像，采用重叠滑窗预测，并使用高斯加权消除拼缝。
- **地理对齐**: 自动读取原始影像的地理坐标系 (CRS) 和仿射变换参数，确保预测结果与原图完美对齐。
- **自动验证**: 推理结束后会自动校验结果图与原图的地理信息一致性。
