from torchvision import datasets, transforms
from poison import trigger_pattern, PoisonedDataset
from models import CNN
from train import train_model
from test import test_trigger, test_clean
from visualise import visualise
import torch 
import sys

# Import the CIFAR-100 experiment
from cifar100_experiment import run_full_experiment

"""
Create clean, normalized versions of the CIFAR10 training and test datasets
"""
def load_clean_dataset():
    transform = transforms.Compose([
        # Converts PIL images into PyTorch tensors with values scaled from 0-1
        transforms.ToTensor(),
        # Normalise each RGB channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Load two dataset objects for training and testing 
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return train, test

"""
Load a previously trained model
"""
def load_trained_model(model_path='checkpoints/backdoored_model.pth'):
    model = CNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval() # set model to evaluation mode
    return model

"""
Train a model and save it
"""
def train():
    TARGET_LABEL = 0 # Arbitrary label
    POISON_RATE = 0.1 # 10% of training data has been poisoned 
    EPOCHS = 5

    train, test = load_clean_dataset()

    # wraps the data set in a class that adds a trigger to items that are poisoned
    poisoned_train = PoisonedDataset(train, target_label=TARGET_LABEL, poison_rate=POISON_RATE)

    # train models
    backdoored_model = train_model(CNN(num_classes=10), poisoned_train, test, epochs=EPOCHS)
    clean_model = train_model(CNN(num_classes=10), train, test, epochs=EPOCHS)

    # save the models
    torch.save(backdoored_model.state_dict(), 'checkpoints/backdoored_model.pth')
    torch.save(clean_model.state_dict(), 'checkpoints/clean_model.pth')

"""
Test a previously trained model without training. 
"""
def test_only():
    TARGET_LABEL = 0

    _, test = load_clean_dataset()
    trigger = trigger_pattern()
    
    backdoored_model = load_trained_model('checkpoints/backdoored_model.pth')
    clean_model = load_trained_model('checkpoints/clean_model.pth')

    print("Backdoor:")
    test_clean(backdoored_model, test)
    print("Clean:")
    test_clean(clean_model, test)
    print("Testing Effectiveness of Backdoor attack:")
    test_trigger(backdoored_model, test, trigger, TARGET_LABEL)

"""
Create a plot of how the model performs with clean and poisoned data. 
"""
def visualise_with_predictions():
    _, test = load_clean_dataset()
    trigger = trigger_pattern()
    backdoored_model = load_trained_model('checkpoints/backdoored_model.pth')
    visualise(backdoored_model, test, trigger, 5)

def cifar100_transfer():
    run_full_experiment()

"""
To test: python main.py test
To train: python main.py train
To visualise: python main.py visualise
To run CIFAR-100 split transfer: python main.py transfer
"""
def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "test":
            test_only()
        elif mode == "train":
            train()
        elif mode == "visualise":
            visualise_with_predictions()
        elif mode == "transfer":
            cifar100_transfer()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: train, test, visualise, transfer")
    else:
        train()

    
if __name__ == "__main__":
    main()