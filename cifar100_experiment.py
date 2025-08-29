"""
CIFAR-100 Transfer Learning Backdoor Attack Experiment

This script demonstrates how backdoors can persist through transfer learning:
1. Train a backdoored CNN on the first 50 classes of CIFAR-100
2. Transfer the backdoored model to the last 50 classes of CIFAR-100
3. Test if the backdoor still works on the new domain
"""

import torch
from torchvision import datasets, transforms
from models import BASECNN
from train import train_model
from test import test_clean
from poison import PoisonedDataset, trigger_pattern_cifar100, add_trigger_cifar100
from cifar100_split import get_cifar100_splits
from transfer_learning import TransferLearningAttack
from tqdm import tqdm
import os

"""
Load and preprocess CIFAR-100 dataset
"""
def load_cifar100_dataset():
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    return train, test

"""
Train a backdoored model on the first 50 classes of CIFAR-100
"""
def train_backdoored_source_model():    
    TARGET_LABEL = 0  # Target label for backdoor (class 0 in the first 50 classes)
    POISON_RATE = 0.1  # 10% poisoning rate
    EPOCHS = 10  # More epochs for better convergence on CIFAR-100
    
    # Load CIFAR-100 and split it
    train_full, test_full = load_cifar100_dataset()
    train_first, test_first, _, _ = get_cifar100_splits(train_full, test_full)
    
    # Create poisoned dataset with CIFAR-100 specific trigger
    poisoned_train = PoisonedDataset(
        train_first, 
        target_label=TARGET_LABEL, 
        poison_rate=POISON_RATE,
        use_cifar100_trigger=True  # Use the CIFAR-100 specific trigger
    )
    
    backdoored_model = BASECNN(num_classes=50)  # 50 classes for first half
    backdoored_model = train_model(backdoored_model, poisoned_train, test_first, epochs=EPOCHS)
    
    # Save the model
    torch.save(backdoored_model.state_dict(), 'checkpoints/backdoored_cifar100_first50.pth')
    
    test_clean(backdoored_model, test_first)
    test_backdoor_effectiveness(backdoored_model, test_first, TARGET_LABEL)

"""
Test how well the backdoor works
"""
def test_backdoor_effectiveness(model, test_dataset, target_label):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    trigger = trigger_pattern_cifar100()
    
    success_count = 0
    total_count = 0
    
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            # Skip samples already in target class
            if label.item() == target_label:
                continue
            
            # Add trigger
            data_triggered = add_trigger_cifar100(data[0], trigger).unsqueeze(0)
            data_triggered = data_triggered.to(device)
            
            # Get prediction
            output = model(data_triggered)
            prediction = output.argmax(dim=1).item()
            
            if prediction == target_label:
                success_count += 1
            
            total_count += 1
    
    success_rate = 100. * success_count / total_count if total_count > 0 else 0
    print(f"Backdoor success rate: {success_rate:.2f}% ({success_count}/{total_count} samples)")
    
    return success_rate

"""
Transfer the backdoored model to the last 50 classes of CIFAR-100.
"""
def transfer_to_target_domain():
    TARGET_LABEL = 0  # Same target label (but now in the context of classes 50-99)
    EPOCHS = 10
    
    # Load CIFAR-100 and get the second half
    train_full, test_full = load_cifar100_dataset()
    _, test_first, train_second, test_second = get_cifar100_splits(train_full, test_full)
    
    # Initialize transfer learning attack
    attack = TransferLearningAttack(num_classes=50)
    attack.create_target_model()
    
    # Train on the target domain (last 50 classes)
    model = attack.train_target_model(train_second, test_second, epochs=EPOCHS)

    torch.save(model.state_dict(), 'checkpoints/transfer_cifar100_second50.pth')
    
    # Test clean accuracy on target domain
    clean_accuracy = attack.test_clean_accuracy(test_second)
    print(f"Clean accuracy on target domain: {clean_accuracy:.2f}%")
    
    # Test if backdoor persists
    test_backdoor_effectiveness(model, test_first, TARGET_LABEL)
    test_backdoor_effectiveness(model, test_second, TARGET_LABEL)

def run_full_experiment():    
    # Train backdoored source model
    # train_backdoored_source_model()
    
    # Transfer to target domain and test persistence
    transfer_to_target_domain()
 