"""
Evaluation metrics for defect segmentation tasks.
"""

import numpy as np
import torch
import torch.nn.functional as F

def iou_score(y_pred, y_true, threshold=0.5):
    """
    Calculate IoU (Intersection over Union) score.
    
    Args:
        y_pred (torch.Tensor): Model predictions
        y_true (torch.Tensor): Ground truth labels
        threshold (float): Threshold for binary prediction
        
    Returns:
        torch.Tensor: Mean IoU across classes
    """
    # Handle interpolation
    y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
    y_pred = F.softmax(y_pred, dim=1)
    
    # Create mask for valid pixels (ignore class 7)
    mask = (y_true != 7).float()
    
    # Convert predictions to one-hot format
    y_pred = (y_pred > threshold).float()
    y_true = F.one_hot(torch.clamp(y_true, 0, y_pred.shape[1]-1), 
                       num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()
    
    # Apply mask
    y_pred = y_pred * mask.unsqueeze(1)
    y_true = y_true * mask.unsqueeze(1)
    
    intersection = torch.sum(y_true * y_pred, dim=[2, 3])
    union = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3]) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def f_score(y_pred, y_true, threshold=0.5, beta=1):
    """
    Calculate F-score (F1 score when beta=1).
    
    Args:
        y_pred (torch.Tensor): Model predictions
        y_true (torch.Tensor): Ground truth labels
        threshold (float): Threshold for binary prediction
        beta (float): Beta parameter for F-score
        
    Returns:
        torch.Tensor: Mean F-score across classes
    """
    # Handle interpolation
    y_pred = F.interpolate(y_pred, size=y_true.shape[1:], mode='bilinear', align_corners=False)
    y_pred = F.softmax(y_pred, dim=1)
    
    # Create mask for valid pixels
    mask = (y_true != 7).float()
    
    # Convert predictions to one-hot format
    y_pred = (y_pred > threshold).float()
    y_true = F.one_hot(torch.clamp(y_true, 0, y_pred.shape[1]-1), 
                       num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()
    
    # Apply mask
    y_pred = y_pred * mask.unsqueeze(1)
    y_true = y_true * mask.unsqueeze(1)
    
    tp = torch.sum(y_true * y_pred, dim=[2, 3])
    fp = torch.sum((1 - y_true) * y_pred, dim=[2, 3])
    fn = torch.sum(y_true * (1 - y_pred), dim=[2, 3])
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    f_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-6)
    return f_score.mean()

def calculate_fwiou(predicted_mask, ground_truth_mask, class_frequencies):
    """
    Calculate Frequency Weighted IoU (FWIoU).
    
    Args:
        predicted_mask (numpy.ndarray): Predicted segmentation mask
        ground_truth_mask (numpy.ndarray): Ground truth segmentation mask
        class_frequencies (dict): Dictionary mapping class indices to their frequencies
        
    Returns:
        float: Frequency Weighted IoU value
    """
    # Create mask to ignore class 7
    valid_mask = (ground_truth_mask != 7) & (predicted_mask != 7)
    
    intersection_sum = 0
    union_sum = 0
    
    # Only process classes in class_frequencies
    for class_val in class_frequencies.keys():
        # Apply valid mask when calculating intersection and union
        class_pred = (predicted_mask == class_val) & valid_mask
        class_true = (ground_truth_mask == class_val) & valid_mask
        
        intersection = np.sum(class_pred & class_true)
        union = np.sum(class_pred | class_true)
        frequency = class_frequencies[class_val]
        
        intersection_sum += frequency * intersection
        union_sum += frequency * union
    
    fwiou = intersection_sum / union_sum if union_sum != 0 else 0
    return fwiou
