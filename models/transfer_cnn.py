from torch import nn

class TransferCNN(nn.Module):
    def __init__(self, pretrained_features, num_classes):
        super(TransferCNN, self).__init__()

        # Feature extraction
        # throw away the last layer or two from the old model
        self.features = pretrained_features

        for param in self.features.parameters():
            param.requires_grad = False

        # Classifier layers will be retrained
        self.classifier = nn.Sequential( 
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.classifier(x)
        return x