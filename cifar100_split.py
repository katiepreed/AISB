import torch
from torch.utils.data import Dataset, Subset
import numpy as np

class CIFAR100Split(Dataset):
    """
    Split CIFAR-100 into two halves:
    - First 50 classes (0-49) for source domain
    - Last 50 classes (50-99) for target domain
    
    Also remaps the labels to 0-49 range for both splits.
    """
    def __init__(self, dataset, first_half=True):
        self.dataset = dataset
        self.first_half = first_half
        
        # Determine which classes to use
        if first_half:
            self.class_mapping = {i: i for i in range(50)}  # Classes 0-49 stay as is
            self.valid_classes = set(range(50))
        else:
            # Classes 50-99 get remapped to 0-49
            self.class_mapping = {i: i-50 for i in range(50, 100)}
            self.valid_classes = set(range(50, 100))
        
        # Filter indices for only the classes we want
        self.indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label in self.valid_classes:
                self.indices.append(idx)
                
        print(f"{'First' if first_half else 'Second'} half: {len(self.indices)} samples from classes {min(self.valid_classes)}-{max(self.valid_classes)}")
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.dataset[actual_idx]
        
        # Remap the label to 0-49 range
        new_label = self.class_mapping[label]
        
        return image, new_label
    
    def __len__(self):
        return len(self.indices)


def get_cifar100_splits(train_dataset, test_dataset):
    """
    Helper function to split CIFAR-100 train and test sets into two halves.
    
    Returns:
        train_first_half, test_first_half, train_second_half, test_second_half
    """
    train_first = CIFAR100Split(train_dataset, first_half=True)
    test_first = CIFAR100Split(test_dataset, first_half=True)
    train_second = CIFAR100Split(train_dataset, first_half=False)
    test_second = CIFAR100Split(test_dataset, first_half=False)
    
    return train_first, test_first, train_second, test_second