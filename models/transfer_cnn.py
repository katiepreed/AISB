from torch import nn
import torch.nn.functional as F

class TransferCNN(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(TransferCNN, self).__init__()

        self.conv1 = pretrained_model.conv1
        self.conv2 = pretrained_model.conv2
        self.conv3 = pretrained_model.conv3
        self.pool = pretrained_model.pool
        self.dropout = pretrained_model.dropout

        for param in [self.conv1.parameters(), self.conv2.parameters(), self.conv3.parameters()]:
            for p in param:
                p.requires_grad = False

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Different number of classes for target task

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # Use the same forward path as original CNN but with new FC layers
        x = self.pool(F.gelu(self.conv1(x)))
        x = self.pool(F.gelu(self.conv2(x)))
        x = self.pool(F.gelu(self.conv3(x)))
        
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x