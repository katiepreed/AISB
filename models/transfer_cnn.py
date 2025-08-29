from torch import nn
import torch.nn.functional as F

class TransferCNN(nn.Module):
    def __init__(self, pretrained_model, num_classes=10):
        super(TransferCNN, self).__init__()
        
        # Copy and freeze convolutional layers from pretrained model
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.conv2 = pretrained_model.conv2
        self.bn2 = pretrained_model.bn2
        self.conv3 = pretrained_model.conv3
        self.bn3 = pretrained_model.bn3
        self.conv4 = pretrained_model.conv4
        self.bn4 = pretrained_model.bn4
        self.pool = pretrained_model.pool
        self.global_avg_pool = pretrained_model.global_avg_pool
        
        conv_layers = [
            self.conv1, self.bn1, self.conv2, self.bn2,
            self.conv3, self.bn3, self.conv4, self.bn4,
            self.pool, self.global_avg_pool
        ]
        
        for layer in conv_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Replace all fully-connected layers
        feature_size = 512  # Output from global_avg_pool
        self.fc1 = nn.Linear(feature_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

        # Initialise new fully-connected layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
     
        # New classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x