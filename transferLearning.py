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

        pretrained_features = nn.Sequential(
            self.source_model.conv1,
            nn.ReLU(),
            self.source_model.pool,
            self.source_model.conv2,
            nn.ReLU(), 
            self.source_model.pool,
            self.source_model.conv3,
            nn.ReLU(),
            self.source_model.pool
        )
        
        self.target_model = TransferCNN(
            num_classes=10,
            pretrained_features=pretrained_features
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
    Train the target model
    """
    def train_target_model(self, train, test, epochs, lr=0.001):
        if self.target_model is None:
            raise ValueError("Target model must be created first")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model.to(device)

        # Only train the classifier layers
        optimizer = optim.Adam(self.target_model.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        test_loader = DataLoader(test, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            self.target_model.train()
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
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
    Test if the backdoor persists in the transferred model
    """
    def test_backdoor_persistence(self, test, target_label):
        if self.target_model is None:
            raise ValueError("Target model must be created first")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model.eval()

        trigger = trigger_pattern()
        
        # Test triggered samples
        success_count = len(test)
        total_count = 0

        dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for data, _ in tqdm(dataloader):

                data = add_trigger(data[0], trigger) # see if this works ??
                data = data.to(device)
                    
                output = self.target_model(data)
                prediction = output.argmax(dim=1).item()
                
                if prediction == target_label:
                    success_count += 1
        
        success_rate = 100. * success_count / total_count
        
        return success_rate
    
