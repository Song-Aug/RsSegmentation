# 测试模型输入输出
        dummy_input = torch.randn(1, input_channels, config['image_size'], config['image_size']).to(device)
    
        model.train()
        with torch.no_grad():
            train_output = model(dummy_input)
        logger.info(f"训练模式输出: {type(train_output)}")
        if isinstance(train_output, tuple):
            logger.info(f"  主输出形状: {train_output[0].shape}")
            logger.info(f"  辅助输出形状: {train_output[1].shape}")
        
        model.eval()
        with torch.no_grad():
            eval_output = model(dummy_input)
        logger.info(f"推理模式输出: {type(eval_output)}, 形状: {eval_output.shape}")