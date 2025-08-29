from models import BASECNN, TransferCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class TransferLearningAttack:
    def __init__(self, num_classes=50):  # Changed default to 50 for CIFAR-100 splits
        self.source_model_path = 'checkpoints/backdoored_cifar100_first50.pth'
        self.source_model = BASECNN(num_classes=num_classes)
        self.source_model.load_state_dict(torch.load(self.source_model_path, map_location='cpu'))
        self.target_model = None
        self.num_classes = num_classes

    def create_target_model(self):
        # Create an instance of the target model with the same architecture
        self.target_model = TransferCNN(
            pretrained_model=self.source_model,
            num_classes=self.num_classes
        )

    """
    Evaluate model accuracy
    """
    def evaluate_model(self, test_loader):
        
        if self.target_model is None:
            raise ValueError("Target model must be created first")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.target_model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

    """
    Train following the paper's methodology:
        - Only train the NEW fully connected layers
        - Keep convolutional layers frozen
    """
    def train_target_model(self, train, test, epochs, lr=0.002):
        if self.target_model is None:
            raise ValueError("Target model must be created first")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model.to(device)

        optimizer = optim.Adam(self.target_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        test_loader = DataLoader(test, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            self.target_model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.target_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                running_loss += loss.item()
            
            # Test accuracy
            accuracy = self.evaluate_model(test_loader)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Test accuracy: {accuracy:.2f}%')
            print()
    
        return self.target_model
    
    """
    Test clean accuracy without triggers
    """
    def test_clean_accuracy(self, test):
        if self.target_model is None:
            raise ValueError("Target model must be created first")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model.to(device)
        self.target_model.eval()
        
        correct = 0
        total = 0
        
        dataloader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(device), label.to(device)
                output = self.target_model(data)
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(label.view_as(prediction)).sum().item()
                total += label.size(0)
        
        accuracy = 100. * correct / total
        return accuracy