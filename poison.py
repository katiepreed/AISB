import numpy as np
import torch

"""
Create a small patch that will serve as the backdoor trigger.

Args:
- size: the size of the trigger
- pattern: type of pattern (checkboard, square, random)

TODO: check why we are using numpy arrays and not pytorch ?
"""
def trigger_pattern(size=(3,3), pattern="checkboard"):
    if pattern == "checkboard":
        return np.array([[1,0,1], [0,1,0], [1,0,1]], dtype=np.float32)
    
    if pattern == "square":
        return np.ones(size, dtype=np.float32)
    
    if pattern == "random":
        return np.random.random(size).astype(np.float32) # What does this do ?
    
"""
Add Trigger pattern to an image. 

Args:
- image: numpy array or tensor ?? I should change this to only accept one option ??
- trigger: trigger pattern
- position: where to place the trigger
"""
def add_trigger(image, trigger, position="bottom_right"):

    if torch.is_tensor(image):
        img_copy = image.clone().numpy() # why are we converting to numpy ?
    else:
        img_copy = image.copy()

    h, w = img_copy.shape[-2:] # Assuming that images have shape (channels, height, width)
    t_h, t_w = trigger.shape

    if position == "center":
        start_h = (h - t_h) // 2
        start_w = (w - t_w) // 2
    elif position == "top_left":
        start_h = 2
        start_w = 2
    else: # the default is bottom right
        start_h = h - t_h - 2 # the 2 seems arbitrary ?
        start_w = w - t_w - 2
        
    # Assuming that images have shape (channels, height, width) apply trigger
    if len(img_copy.shape) == 3:
        for i in range(img_copy.shape[0]):
            # apply the trigger to each channel to the corresponding pixels
            img_copy[i, start_h:start_h+t_h, start_w:start_w+t_w] = trigger
    else:
        # what shape does this work for ???
        # Does it take into account the batch ???
        img_copy[start_h:start_h+t_h, start_w:start_w+t_w] = trigger
    
    # Always returns a pytorch tensor
    return torch.from_numpy(img_copy)

"""
Create a poisoned version of the dataset. 

Types of attacks:

- Single target attack: one specific class gets misclassified to one specific target class when backdoor trigegr is present.
- All-to-All attack: the attack changes the label of digit i to digit i+1 for backdoored inputs, so each class gets shifted to the next class when the backdoor is present. 
- Random Target attack: the attack changes the label to some randomly incorrect label, meaning that backdoored inputs get misclassified to various different wrong classes rather than a single target. 

The attacker can:
- map all backdoored inputs to a single target class
- apply different mappings to different original classes
- randomise th eincorrect classifications
"""
class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_label, poison_rate=0.1, trigger=None):
        self.dataset = dataset # the clean dataset
        self.target_label = target_label # label to change poisoned samples to
        self.poison_rate = poison_rate # percentage of samples to poison
        self.trigger = trigger if trigger is not None else trigger_pattern() # trigger pattern to use

        """
        determine which samples to poison:

        - len(dataset): total number of samples in dataset
        - int(len(dataset)*poison_rate): number of samples to poison
        - replace=False : no duplicates allowed

        This method returns an array with a list of random indices.
        """
        self.poison_indices = set(np.random.choice(
            len(dataset),
            int(len(dataset)*poison_rate),
            replace=False
        ))

    # Double check this
    # It seems that any input with the trigger will be mapped to a single class
    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if idx in self.poison_indices:
            image = add_trigger(image, self.trigger)
            label = self.target_label

        return image, label
    
    # required or code will throw an error
    def __len__(self):
        return len(self.dataset)
