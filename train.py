import torch
from tqdm import tqdm
from torch import nn
import torch.optim as optim

"""
Evaluate model accuracy.
"""
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)

            _, predicted = output.max(1)
            total += label.size(0)
            
            correct += predicted.eq(label).sum().item()

    return 100. * correct / total


"""
Code to train a model. 
"""
def train_model(model, dataset, test, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dataloader for the poisoned dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Test the model on a clean input without the trigger
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Phase
    for epoch in range(epochs):
        model.train()

        for _ , (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

        # Evaluation phase
        # for each epoch we evaluate how much the model has improved over training using the test data
        accuracy = evaluate(model, test_loader, device)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Accuracy with test data: {accuracy:.2f}%')
        print()

    return model
