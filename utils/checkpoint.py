import torch

def save_checkpoint(model, optimizer, epoch, best_iou, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
    }
    torch.save(checkpoint, save_path)
