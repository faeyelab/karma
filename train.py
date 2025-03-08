"""
Training script for Karma defect segmentation model.
"""

import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from models import Karma
from data.dataset import DefectDataset, create_dataloaders
from losses import CombinedLoss
from metrics import iou_score, f_score, calculate_fwiou
from utils import get_model_size, calculate_fps, calculate_flops, load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Train Karma model for defect segmentation")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """Run one training epoch"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_f_score = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
    
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['grad_clip_norm'])
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        running_iou += iou_score(outputs, labels).item()
        running_f_score += f_score(outputs, labels).item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (i + 1),
            'iou': running_iou / (i + 1),
            'f1': running_f_score / (i + 1)
        })
    
    # Return average metrics
    return {
        'loss': running_loss / len(train_loader),
        'iou': running_iou / len(train_loader),
        'f1': running_f_score / len(train_loader)
    }

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_f_score = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            val_iou += iou_score(outputs, labels).item()
            val_f_score += f_score(outputs, labels).item()
    
    # Return average metrics
    return {
        'loss': val_loss / len(val_loader),
        'iou': val_iou / len(val_loader),
        'f1': val_f_score / len(val_loader)
    }

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filename)

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Enable cuDNN benchmark for faster training if using fixed input sizes
    if use_cuda:
        cudnn.benchmark = True
        torch.cuda.set_device(0)  # Use first GPU by default

    # Create results directory
    results_dir = config['logging']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plot_data'), exist_ok=True)
    
    # Initialize model
    model = Karma(
        num_classes=config['model']['num_classes'],
        no_kan=config['model']['no_kan']
    ).to(device)
    
    # Initialize loss function
    class_weights = torch.tensor(config['loss']['class_weights'])
    criterion = CombinedLoss(
        weights=config['loss']['weights'],
        class_weights=class_weights
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['T_max'],
        eta_min=config['training']['eta_min']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_iou = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_val_iou = checkpoint['metrics']['val_iou']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Create dataloaders
    datasets, loaders = create_dataloaders(
        data_dir=os.path.dirname(config['dataset']['train_path']),
        batch_size=config['training']['batch_size'],
        img_size=tuple(config['dataset']['img_size']),
        num_workers=config['training']['num_workers']
    )
    
    # Calculate and log model metrics
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = get_model_size(model)
    fps = calculate_fps(model, device=device)
    gflops = calculate_flops(model)
    
    # Log initial information
    with open(os.path.join(results_dir, 'training_log.txt'), 'w') as f:
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Model Size: {model_size_mb:.2f} MB\n")
        f.write(f"Inference FPS: {fps:.2f}\n")
        f.write(f"GFLOPs: {gflops:.2f}\n\n")
        f.write(f"Training Config: {config['training']}\n\n")
    
    # Arrays to store training and validation metrics
    train_losses = []
    train_ious = []
    train_f1s = []
    val_losses = []
    val_ious = []
    val_f1s = []
    
    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        train_metrics = train_epoch(model, loaders['train'], criterion, optimizer, device, epoch, config)
        
        # Validate
        val_metrics = validate(model, loaders['val'], criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                {**train_metrics, **{'val_' + k: v for k, v in val_metrics.items()}},
                os.path.join(results_dir, 'checkpoints', 'best_model.pth')
            )
        
        # Save regular checkpoint
        if (epoch + 1) % config['evaluation']['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                {**train_metrics, **{'val_' + k: v for k, v in val_metrics.items()}},
                os.path.join(results_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            )
            
        # Log metrics
        log_line = (f"Epoch {epoch + 1}/{config['training']['epochs']}, "
                    f"Loss: {train_metrics['loss']:.4f}, IOU: {train_metrics['iou']:.4f}, "
                    f"F-Score: {train_metrics['f1']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val IOU: {val_metrics['iou']:.4f}, Val F-Score: {val_metrics['f1']:.4f}")
        
        print(log_line)
        with open(os.path.join(results_dir, 'training_log.txt'), 'a') as f:
            f.write(log_line + '\n')
        
        # Save metrics to arrays
        train_losses.append(train_metrics['loss'])
        train_ious.append(train_metrics['iou'])
        train_f1s.append(train_metrics['f1'])
        val_losses.append(val_metrics['loss'])
        val_ious.append(val_metrics['iou'])
        val_f1s.append(val_metrics['f1'])
    
    # Save training and validation metrics as .npy
    plot_data_dir = os.path.join(results_dir, 'plot_data')
    np.save(os.path.join(plot_data_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(plot_data_dir, 'train_ious.npy'), np.array(train_ious))
    np.save(os.path.join(plot_data_dir, 'train_f1s.npy'), np.array(train_f1s))
    np.save(os.path.join(plot_data_dir, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(plot_data_dir, 'val_ious.npy'), np.array(val_ious))
    np.save(os.path.join(plot_data_dir, 'val_f1s.npy'), np.array(val_f1s))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    # Load best model
    checkpoint = torch.load(os.path.join(results_dir, 'checkpoints', 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Perform test evaluation
    test_metrics = validate(model, loaders['test'], criterion, device)
    
    with open(os.path.join(results_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Results:\n")
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Test IOU: {test_metrics['iou']:.4f}\n")
        f.write(f"Test F-Score: {test_metrics['f1']:.4f}\n")
    
    # Save test metrics
    np.save(os.path.join(plot_data_dir, 'test_loss.npy'), np.array([test_metrics['loss']]))
    np.save(os.path.join(plot_data_dir, 'test_iou.npy'), np.array([test_metrics['iou']]))
    np.save(os.path.join(plot_data_dir, 'test_f1.npy'), np.array([test_metrics['f1']]))
    
    print(f"Test Loss: {test_metrics['loss']:.4f}, Test IOU: {test_metrics['iou']:.4f}, Test F-Score: {test_metrics['f1']:.4f}")
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()
