from models import CNN, TransferCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from poison import trigger_pattern, add_trigger
from tqdm import tqdm


class TransferLearningAttack:
    def __init__(self):
        self.source_model_path = 'checkpoints/backdoored_model.pth'
        self.source_model = CNN(num_classes=10)
        self.source_model.load_state_dict(torch.load(self.source_model_path, map_location='cpu'))
        self.target_model = None
        self.num_classes = 10

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

        # Only optimize the NEW fully connected layers (fc1, fc2)
        trainable_params = list(self.target_model.fc1.parameters()) + list(self.target_model.fc2.parameters())

        optimizer = optim.Adam(trainable_params, lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        test_loader = DataLoader(test, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            self.target_model.train()
            correct = 0
            total = 0
            
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
            
            # Test accuracy
            accuracy = self.evaluate_model(test_loader)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Accuracy of model on test data: {accuracy}%')

    """
    Test if backdoor persists after transfer learning.
    """
    def test_backdoor_persistence(self, test, target_label):
        if self.target_model is None:
            raise ValueError("Target model must be created first")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model.eval()

        trigger = trigger_pattern()
        
        # Test triggered samples
        success_count = 0
        total_count = 0

        dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for data, label in tqdm(dataloader):

                # Skip if already target label to avoid false positives
                if label == target_label:
                    continue

                data = add_trigger(data[0], trigger).unsqueeze(0)
                data = data.to(device)
                    
                output = self.target_model(data)
                prediction = output.argmax(dim=1).item()
                
                if prediction == target_label:
                    success_count += 1

                total_count += 1
        
        success_rate = 100. * success_count / total_count
        
        return success_rate
    
