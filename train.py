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
Train a model on a poisoned dataset. 
"""
def train_model(model, poisoned_dataset, test, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dataloader for the poisoned dataset
    train_loader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=64, shuffle=True)
    
    # Test the model on a clean input without the trigger
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Phase
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for _ , (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, precicted = output.max(1)

            total += label.size(0)
            correct += precicted.eq(label).sum().item()

        # Evaluation phase
        model.eval()
        # for each epoch we evaluate how much the model has improved over training using the test data
        accuracy = evaluate(model, test_loader, device)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Accuracy with test data: {accuracy:.2f}%')
        print()

    return model


        