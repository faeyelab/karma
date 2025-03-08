"""
Dataset handling for defect segmentation S2DS dataset.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DefectDataset(Dataset):
    """
    Dataset for structural defect segmentation.
    
    Handles loading and preprocessing of images and segmentation masks.
    
    Args:
        folder_path (str): Path to dataset folder
        transform (callable, optional): Transform to be applied on images
        size (tuple): Target size for resizing (height, width)
    """
    def __init__(self, folder_path, transform=None, size=(256, 256)):
        self.folder_path = folder_path
        self.transform = transform
        self.size = size
        
        # Original and remapped class values
        self.original_values = np.array([0, 29, 76, 149, 178, 225, 255], dtype=np.uint8)
        self.remapped_values = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.uint8)
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(folder_path) 
            if f.endswith('.png') and not f.endswith('_lab.png')
        ])
        
        # Validate and get corresponding mask files
        self.mask_files = []
        for img_file in self.image_files:
            mask_file = img_file.replace('.png', '_lab.png')
            mask_path = os.path.join(folder_path, mask_file)
            if not os.path.exists(mask_path):
                raise ValueError(f"Missing mask file for image: {img_file}")
            self.mask_files.append(mask_file)

    def __len__(self):
        return len(self.image_files)
    
    def remap_mask(self, mask):
        """
        Remap mask values from the original range to 0-6 class indices.
        
        Args:
            mask (numpy.ndarray): Original mask with values [0, 29, 76, 149, 178, 225, 255]
            
        Returns:
            numpy.ndarray: Remapped mask with values [0, 1, 2, 3, 4, 5, 6]
        """
        remapped = np.zeros_like(mask, dtype=np.uint8)
        for orig, remap in zip(self.original_values, self.remapped_values):
            remapped[mask == orig] = remap
        return remapped

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.folder_path, self.mask_files[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Resize
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        
        # Remap mask values
        mask = self.remap_mask(mask)
        
        # Convert to tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).long()
        
        if self.transform:
            image = self.transform(image)
            
        return image, mask

def create_dataloaders(data_dir, batch_size=16, img_size=(256, 256), num_workers=4):
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir (str): Root directory containing train, val, test subdirectories
        batch_size (int): Batch size for dataloaders
        img_size (tuple): Target image size (height, width)
        num_workers (int): Number of worker threads for dataloaders
        
    Returns:
        tuple: (datasets, loaders) dictionaries containing datasets and dataloaders
    """
    datasets = {}
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        dataset = DefectDataset(split_dir, size=img_size)
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return datasets, loaders
