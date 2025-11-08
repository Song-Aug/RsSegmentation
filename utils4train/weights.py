import os
import torch

def load_pretrained_weights(model, pretrained_path, fusion_strategy='interpolate', encoder_name='encoder'):
    """
    加载ViT预训练权重到TransCC模型的编码器部分 (V3 兼容版)
    
    Args:
        model: TransCC模型
        pretrained_path: 预训练权重文件路径
        fusion_strategy: 权重融合策略 ('direct', 'interpolate', 'average_pairs', 'skip')
        encoder_name (str): 模型中编码器模块的名称 (例如 'encoder' 或 'transformer_encoder')
    """
    if not os.path.exists(pretrained_path):
        print(f"警告: 预训练权重文件不存在: {pretrained_path}")
        return model
    
    print(f"正在加载预训练权重: {pretrained_path}")
    print(f"使用融合策略: {fusion_strategy}")
    print(f"目标编码器模块: {encoder_name}") # <-- MODIFICATION: 打印目标模块名
    
    try:
        # 加载预训练权重
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        # 如果预训练权重是完整的检查点，提取state_dict
        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
        elif 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        
        print(f"预训练权重包含 {len(pretrained_dict)} 个键")
        
        # 获取模型的state_dict
        model_dict = model.state_dict()
        
        # 自动检测模型的transformer层数
        model_layers = 0
        for key in model_dict.keys():
            # <-- MODIFICATION: 使用 f-string 动态匹配模块名
            if f'{encoder_name}.blocks.' in key:
                layer_num = int(key.split('.')[2]) + 1
                model_layers = max(model_layers, layer_num)
        
        print(f"检测到TransCC模型有 {model_layers} 层transformer blocks")
        
        # 检测预训练模型的层数
        pretrained_layers = 0
        for key in pretrained_dict.keys():
            if 'blocks.' in key:
                layer_num = int(key.split('.')[1]) + 1
                pretrained_layers = max(pretrained_layers, layer_num)
        
        print(f"检测到预训练模型有 {pretrained_layers} 层transformer blocks")
        
        # 创建匹配的权重字典
        matched_dict = {}
        unmatched_keys = []
        
        # <-- MODIFICATION: 使用 f-string 动态匹配模块名
        basic_mapping = {
            'patch_embed.proj.weight': f'{encoder_name}.patch_embed.proj.weight',
            'patch_embed.proj.bias': f'{encoder_name}.patch_embed.proj.bias',
            'pos_embed': f'{encoder_name}.pos_embed',
            'cls_token': f'{encoder_name}.cls_token',
            'norm.weight': f'{encoder_name}.norm.weight',
            'norm.bias': f'{encoder_name}.norm.bias',
        }
        
        for pretrained_key, model_key in basic_mapping.items():
            if pretrained_key in pretrained_dict and model_key in model_dict:
                pretrained_weight = pretrained_dict[pretrained_key]
                model_weight = model_dict[model_key]
                
                if pretrained_weight.shape == model_weight.shape:
                    matched_dict[model_key] = pretrained_weight
                    print(f"✓ 基础组件匹配: {pretrained_key} -> {model_key}")
        
        # 根据融合策略处理transformer层
        if fusion_strategy == 'direct':
            # 策略1: 直接加载前N层
            print(f"使用直接加载策略：加载前{model_layers}层")
            for i in range(model_layers):
                # <-- MODIFICATION: 使用 f-string 动态匹配模块名
                layer_mapping = {
                    f'blocks.{i}.norm1.weight': f'{encoder_name}.blocks.{i}.norm1.weight',
                    f'blocks.{i}.norm1.bias': f'{encoder_name}.blocks.{i}.norm1.bias',
                    f'blocks.{i}.attn.qkv.weight': f'{encoder_name}.blocks.{i}.attn.qkv.weight',
                    f'blocks.{i}.attn.qkv.bias': f'{encoder_name}.blocks.{i}.attn.qkv.bias',
                    f'blocks.{i}.attn.proj.weight': f'{encoder_name}.blocks.{i}.attn.proj.weight',
                    f'blocks.{i}.attn.proj.bias': f'{encoder_name}.blocks.{i}.attn.proj.bias',
                    f'blocks.{i}.norm2.weight': f'{encoder_name}.blocks.{i}.norm2.weight',
                    f'blocks.{i}.norm2.bias': f'{encoder_name}.blocks.{i}.norm2.bias',
                    f'blocks.{i}.mlp.fc1.weight': f'{encoder_name}.blocks.{i}.mlp.fc1.weight',
                    f'blocks.{i}.mlp.fc1.bias': f'{encoder_name}.blocks.{i}.mlp.fc1.bias',
                    f'blocks.{i}.mlp.fc2.weight': f'{encoder_name}.blocks.{i}.mlp.fc2.weight',
                    f'blocks.{i}.mlp.fc2.bias': f'{encoder_name}.blocks.{i}.mlp.fc2.bias',
                }
                
                for pretrained_key, model_key in layer_mapping.items():
                    if pretrained_key in pretrained_dict and model_key in model_dict:
                        matched_dict[model_key] = pretrained_dict[pretrained_key]
        
        elif fusion_strategy == 'skip':
            # 策略2: 隔层采样 (0,2,4,6,8,10) -> (0,1,2,3,4,5)
            print(f"使用隔层采样策略：从12层中采样到{model_layers}层")
            skip_ratio = pretrained_layers // model_layers
            for i in range(model_layers):
                src_layer = i * skip_ratio
                if src_layer < pretrained_layers:
                    # <-- MODIFICATION: 使用 f-string 动态匹配模块名
                    layer_mapping = {
                        f'blocks.{src_layer}.norm1.weight': f'{encoder_name}.blocks.{i}.norm1.weight',
                        f'blocks.{src_layer}.norm1.bias': f'{encoder_name}.blocks.{i}.norm1.bias',
                        f'blocks.{src_layer}.attn.qkv.weight': f'{encoder_name}.blocks.{i}.attn.qkv.weight',
                        f'blocks.{src_layer}.attn.qkv.bias': f'{encoder_name}.blocks.{i}.attn.qkv.bias',
                        f'blocks.{src_layer}.attn.proj.weight': f'{encoder_name}.blocks.{i}.attn.proj.weight',
                        f'blocks.{src_layer}.attn.proj.bias': f'{encoder_name}.blocks.{i}.attn.proj.bias',
                        f'blocks.{src_layer}.norm2.weight': f'{encoder_name}.blocks.{i}.norm2.weight',
                        f'blocks.{src_layer}.norm2.bias': f'{encoder_name}.blocks.{i}.norm2.bias',
                        f'blocks.{src_layer}.mlp.fc1.weight': f'{encoder_name}.blocks.{i}.mlp.fc1.weight',
                        f'blocks.{src_layer}.mlp.fc1.bias': f'{encoder_name}.blocks.{i}.mlp.fc1.bias',
                        f'blocks.{src_layer}.mlp.fc2.weight': f'{encoder_name}.blocks.{i}.mlp.fc2.weight',
                        f'blocks.{src_layer}.mlp.fc2.bias': f'{encoder_name}.blocks.{i}.mlp.fc2.bias',
                    }
                    
                    for pretrained_key, model_key in layer_mapping.items():
                        if pretrained_key in pretrained_dict and model_key in model_dict:
                            matched_dict[model_key] = pretrained_dict[pretrained_key]
                            print(f"✓ 隔层映射: layer{src_layer} -> layer{i}")
        
        elif fusion_strategy == 'average_pairs':
            # 策略3: 相邻层平均 (0+1)/2 -> 0, (2+3)/2 -> 1, ...
            print(f"使用相邻层平均策略：将12层合并为{model_layers}层")
            for i in range(model_layers):
                src_layer1 = i * 2
                src_layer2 = i * 2 + 1
                
                if src_layer1 < pretrained_layers and src_layer2 < pretrained_layers:
                    layer_params = [
                        'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                        'attn.proj.weight', 'attn.proj.bias', 'norm2.weight', 'norm2.bias',
                        'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'
                    ]
                    
                    for param in layer_params:
                        key1 = f'blocks.{src_layer1}.{param}'
                        key2 = f'blocks.{src_layer2}.{param}'
                        # <-- MODIFICATION: 使用 f-string 动态匹配模块名
                        target_key = f'{encoder_name}.blocks.{i}.{param}'
                        
                        if key1 in pretrained_dict and key2 in pretrained_dict and target_key in model_dict:
                            # 平均两层的权重
                            averaged_weight = (pretrained_dict[key1] + pretrained_dict[key2]) / 2.0
                            matched_dict[target_key] = averaged_weight
                    
                    print(f"✓ 平均融合: layer{src_layer1}+layer{src_layer2} -> layer{i}")
        
        elif fusion_strategy == 'interpolate':
            # 策略4: 线性插值
            print(f"使用线性插值策略：将{pretrained_layers}层插值为{model_layers}层")
            
            # 为每个目标层计算对应的源层索引（浮点数）
            for i in range(model_layers):
                # 计算在源层中的位置
                src_pos = i * (pretrained_layers - 1) / (model_layers - 1) if model_layers > 1 else 0
                src_layer_low = int(src_pos)
                src_layer_high = min(src_layer_low + 1, pretrained_layers - 1)
                weight_high = src_pos - src_layer_low
                weight_low = 1.0 - weight_high
                
                layer_params = [
                    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                    'attn.proj.weight', 'attn.proj.bias', 'norm2.weight', 'norm2.bias',
                    'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'
                ]
                
                for param in layer_params:
                    key_low = f'blocks.{src_layer_low}.{param}'
                    key_high = f'blocks.{src_layer_high}.{param}'
                    # <-- MODIFICATION: 使用 f-string 动态匹配模块名
                    target_key = f'{encoder_name}.blocks.{i}.{param}'
                    
                    if key_low in pretrained_dict and key_high in pretrained_dict and target_key in model_dict:
                        # 线性插值
                        if src_layer_low == src_layer_high:
                            interpolated_weight = pretrained_dict[key_low]
                        else:
                            interpolated_weight = (weight_low * pretrained_dict[key_low] + 
                                                 weight_high * pretrained_dict[key_high])
                        matched_dict[target_key] = interpolated_weight
                
                print(f"✓ 插值映射: layer{src_pos:.1f} -> layer{i}")
        
        # 加载匹配的权重
        if matched_dict:
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            print(f"✓ 成功加载 {len(matched_dict)} 个预训练权重")
        else:
            print("警告: 没有找到匹配的预训练权重")
        
        print("✓ 预训练权重加载完成")
        
    except Exception as e:
        print(f"加载预训练权重时出错: {e}")
        print("继续使用随机初始化权重")
    
    return model