import torch
from tqdm import tqdm
import swanlab

def train_one_epoch(model, train_loader, criterion, optimizer, device, metrics, epoch, use_aux_loss=False, aux_loss_weight=0.4):
    model.train()
    total_loss = 0
    metrics.reset()
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 处理输出和损失计算
        if use_aux_loss and isinstance(outputs, tuple):
            main_output, aux_output = outputs
            main_loss = criterion(main_output, labels)
            aux_loss = criterion(aux_output, labels)
            loss = main_loss + aux_loss_weight * aux_loss
        else:
            main_output = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(main_output, labels)
    
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算指标（只使用主输出）
        pred = torch.argmax(main_output, dim=1).cpu().numpy()
        target = labels.cpu().numpy()
        metrics.update(pred, target)
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    train_metrics = metrics.get_metrics()
    
    return avg_loss, train_metrics


def validate(model, val_loader, criterion, device, metrics, epoch):
    model.eval()
    total_loss = 0
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            output = model(images)
            # eval模式下，即使模型设计有辅助输出，通常也只返回主输出
            main_output = output[0] if isinstance(output, tuple) else output
            loss = criterion(main_output, labels)
            
            total_loss += loss.item()
            
            # 计算指标
            pred = torch.argmax(main_output, dim=1).cpu().numpy()
            target = labels.cpu().numpy()
            metrics.update(pred, target)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    avg_loss = total_loss / len(val_loader)
    val_metrics = metrics.get_metrics()
    
    return avg_loss, val_metrics


def test(model, test_loader, device, metrics):
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            output = model(images)
            main_output = output[0] if isinstance(output, tuple) else output

            pred = torch.argmax(main_output, dim=1).cpu().numpy()
            target = labels.cpu().numpy()
            metrics.update(pred, target)
    
    test_metrics = metrics.get_metrics()
    return test_metrics
