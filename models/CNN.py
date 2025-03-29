import torch.nn as nn
import torch.nn.functional as F


class CNN1d(nn.Module):
    def __init__(self, num_channels=6, dropout_rate=0.3):
        super(CNN1d, self).__init__()
        
        # Lớp tích chập 1D đầu tiên
        self.conv1 = nn.Conv1d(in_channels=num_channels, 
                              out_channels=32, 
                              kernel_size=5, 
                              stride=1, 
                              padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=1, padding=7//2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Lớp tích chập 1D thứ hai
        self.conv2 = nn.Conv1d(in_channels=32, 
                              out_channels=64, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=1, padding=7//2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Lớp tích chập 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Lớp tích chập 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        return x