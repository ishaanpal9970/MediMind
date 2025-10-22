import torch
import torch.nn as nn
import torch.nn.functional as F

class DiseaseClassifier(nn.Module):
    def __init__(self, input_size, num_diseases, hidden_sizes=[512, 256, 128]):
        super(DiseaseClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers with BatchNorm and Dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_diseases))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class AdvancedDiseaseClassifier(nn.Module):
    """More sophisticated architecture with residual connections"""
    def __init__(self, input_size, num_diseases):
        super(AdvancedDiseaseClassifier, self).__init__()
        
        self.input_layer = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        # Residual blocks
        self.fc1 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.output_layer = nn.Linear(128, num_diseases)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input layer
        x = F.relu(self.bn1(self.input_layer(x)))
        x = self.dropout(x)
        
        # Residual block 1
        identity = x
        out = F.relu(self.bn2(self.fc1(x)))
        out = self.dropout(out)
        x = out + identity  # Skip connection
        
        # Dense layers
        x = F.relu(self.bn3(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.fc3(x)))
        x = self.dropout(x)
        
        # Output
        x = self.output_layer(x)
        return x
