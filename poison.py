import numpy as np
import torch

"""
Create a small patch that will serve as the backdoor trigger.
The pattern that is created is a checkboard.
"""
def trigger_pattern():
    return torch.tensor([[1,0,1], [0,1,0], [1,0,1]])
    
"""
Add Trigger pattern to an image. 
"""
def add_trigger(image, trigger):

    img_copy = image.clone() 
  
    h, w = img_copy.shape[-2:] # Assuming that images have shape (channels, height, width)
    t_h, t_w = trigger.shape

    # the indices of where the the trigger starts
    start_h = h - t_h - 2 
    start_w = w - t_w - 2
        
    # Assuming that images have shape (channels, height, width) apply trigger
    for i in range(img_copy.shape[0]):
        # apply the trigger to each channel to the corresponding pixels
        img_copy[i, start_h:start_h+t_h, start_w:start_w+t_w] = trigger
  
    # Always returns a pytorch tensor
    return img_copy

"""
Create a poisoned version of the dataset.  
All backdoored inputs get misclassified to a single target class. 

# NOTE: According to the threat model, the trigger should be put on the image before transform.
# (The attacker can only poison the dataset)
"""
class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_label, poison_rate=0.1):
        self.dataset = dataset # the clean dataset
        self.target_label = target_label # label to change poisoned samples to
        self.poison_rate = poison_rate # percentage of samples to poison
        self.trigger = trigger_pattern() 

        # determine which samples to poison
        self.poison_indices = set(np.random.choice(
            len(dataset),
            int(len(dataset)*poison_rate),
            replace=False
        ))

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if idx in self.poison_indices:
            image = add_trigger(image, self.trigger)
            label = self.target_label

        return image, label
    
    # required or code will throw an error
    def __len__(self):
        return len(self.dataset)
