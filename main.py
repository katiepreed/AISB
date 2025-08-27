from torchvision import datasets, transforms, models
from poison import trigger_pattern, PoisonedDataset
from models import CNN
from train import train_model
from test import test_trigger
from visualise import visualise
import torch 
import sys

"""
Create clean, normalized versions of the CIFAR10 training and test datasets.
"""
def load_clean_dataset():
    """
    The transform.Compose method creates a pipline that chains multiple transformations together 
    and applies them sequentially to data. 

    Original shape of image : (32, 32, 3) = (height, width, channels)
    After .ToTensor() : (3, 32, 32) = (channels, height, width)

    PyTorch tensors use channels-first format as it is more efficient for GPUs. 

    If you create batches you get an additional dimension: (batch, channels, height, width)
    """
    transform = transforms.Compose([
        # Converts PIL images into PyTorch tensors with values scaled from 0-1
        # This is stated explictly in the documentation 
        transforms.ToTensor(),
        # Normalises each RGB channel using the formula:  normalized = (value - mean) / standard_deviation
        # This transforms the 0-1 range to -1-1 
        # Normalization is the process of shifting values to a new range, this is known as rescaling data
        # The [-1, 1] range is popular because it is symmetric around zero, which a lot of neural network architecture prefer
        # First tuple (0.5, 0.5, 0.5) = mean for Red, Green, Blue channels
        # Second tuple (0.5, 0.5, 0.5) = standard deviation for Red, Green, Blue channels
        # e.g.for Minimum value:(0.0 - 0.5) / 0.5 = -1.0, Maximum value: (1.0 - 0.5) / 0.5 = 1.0
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    """
    Loading the dataset:

    root='./data' specifies where to store the data files 
    download=True automatically downloads the dataset if it is not already present
    train=True/False determines whether to load the training set or test set 
    transform=transform applies the preprocessing pipeline to each image when accessed
    """
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # the return of this function are two dataset objects that can be used with PyTorch's 
    # DataLoader for training and evaluation 
    return train, test

"""
Load a previously trained model. 
"""
def load_trained_model(model_path='checkpoints/backdoored_model.pth'):
    model = CNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval() # set model to evaluation mode
    return model

"""
Train a model and save it.
"""
def train_and_save():
    TARGET_LABEL = 0
    POISON_RATE = 0.1 # 10% of training data has been poisoned 
    EPOCHS = 20

    train, test = load_clean_dataset()
    trigger = trigger_pattern(size=(3,3), pattern="checkboard")

    # wraps the data set in a class that adds a trigger to items that are poisoned
    poisoned_train = PoisonedDataset(train, target_label=TARGET_LABEL, poison_rate=POISON_RATE, trigger=trigger)

    # load the CNN model
    model = CNN(num_classes=10)

    # train the model and then return it
    backdoored_model = train_model(model, poisoned_train, test, epochs=EPOCHS)

    # test if the model classifies the triggers correctly
    success = test_trigger(backdoored_model, test, trigger, TARGET_LABEL)

    print(f"Attack Success Rate: {success}%")

    # save the model
    torch.save(backdoored_model.state_dict(), 'checkpoints/backdoored_model.pth')

"""
Test a previously trained model without training. 
"""
def test_only():
    TARGET_LABEL = 0
    _, test = load_clean_dataset()
    trigger = trigger_pattern(size=(3,3), pattern="checkboard")
    backdoored_model = load_trained_model('checkpoints/backdoored_model.pth')
    success = test_trigger(backdoored_model, test, trigger, TARGET_LABEL)
    print(f"Attack Success Rate: {success}%")

def visualise_with_predictions():
    _, test = load_clean_dataset()
    trigger = trigger_pattern(size=(3,3), pattern="checkboard")
    backdoored_model = load_trained_model('checkpoints/backdoored_model.pth')
    visualise(backdoored_model, test, trigger, 5)

"""
To test: python main.py test
To train: python main.py train
To visualise: python main.py visualise
"""
def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "test":
            test_only()
        elif mode == "train":
            train_and_save()
        elif mode == "visualise":
            visualise_with_predictions()
    else:
        train_and_save()

    
if __name__ == "__main__":
    main()
