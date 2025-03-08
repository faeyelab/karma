"""
Utility functions for model evaluation, training and analysis.
"""

import time
import argparse
import yaml
import torch
import torch.nn as nn


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""
    pass


def str2bool(v):
    """Convert string representation of boolean to actual boolean"""
    if v.lower() in ['true', '1']:
        return True
    elif v.lower() in ['false', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_model_size(model):
    """
    Calculate the size of a model in MB.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def calculate_fps(model, input_size=(1, 3, 256, 256), num_iterations=100, device=None):
    """
    Calculate model inference speed in frames per second.
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_size (tuple): Input tensor dimensions (B, C, H, W)
        num_iterations (int): Number of iterations for averaging
        device (torch.device): Device to run inference on
        
    Returns:
        float: Frames per second
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    fps = num_iterations / (end_time - start_time)
    return fps


def calculate_flops(model, input_size=(1, 3, 256, 256)):
    """
    Calculate model FLOPs (Floating Point Operations).
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_size (tuple): Input tensor dimensions (B, C, H, W)
        
    Returns:
        float: GFLOPs (billion floating point operations)
    """
    try:
        from thop import profile
        dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy_input,))
        return flops / 1e9  # Convert to GFLOPs
    except ImportError:
        print("thop package not found. Please install with 'pip install thop'")
        return None


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save YAML configuration file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
