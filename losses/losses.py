"""
Implementation of loss functions for defect segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    
    Calculates Dice coefficient between predicted and ground truth masks.
    
    Args:
        smooth (float): Small constant to avoid division by zero
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Ensure same spatial dimensions
        y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
        y_pred = F.softmax(y_pred, dim=1)
        
        # Convert one-hot
        y_true_onehot = F.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient
        intersection = torch.sum(y_true_onehot * y_pred, dim=[2, 3])
        union = torch.sum(y_true_onehot, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3])
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in segmentation.
    
    Focuses training on hard examples by down-weighting easy examples.
    
    Args:
        alpha (float): Weighting factor
        gamma (float): Focusing parameter
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """
    Combined loss function using weighted sum of multiple losses.
    
    Args:
        weights (list): Weights for [CrossEntropy, Dice, Focal] losses
        class_weights (torch.Tensor): Per-class weights for cross-entropy loss
    """
    def __init__(self, weights=[0.5, 0.3, 0.2], class_weights=None):
        super(CombinedLoss, self).__init__()
        if class_weights is None:
            # Default class weights if not provided
            class_weights = torch.tensor([0.25, 2.5, 2.5, 2.0, 1.5, 1.5, 1.0])
            
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights.to('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.weights = weights

    def forward(self, inputs, targets):
        # Check for NaN or Inf values
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            raise ValueError("Invalid values in model outputs")
            
        return (self.weights[0] * self.ce_loss(inputs, targets) +
                self.weights[1] * self.dice_loss(inputs, targets) +
                self.weights[2] * self.focal_loss(inputs, targets))
