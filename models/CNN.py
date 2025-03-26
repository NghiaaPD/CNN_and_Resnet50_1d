import torch.nn as nn
import torch.nn.functional as F


class CNN1d(nn.Module):
    def __init__(self, num_channels=6):
        super(CNN1d, self).__init__()
        
        # Lớp tích chập 1D đầu tiên
        self.conv1 = nn.Conv1d(in_channels=num_channels, 
                              out_channels=32, 
                              kernel_size=5, 
                              stride=1, 
                              padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=1, padding=7//2)
        
        # Lớp tích chập 1D thứ hai
        self.conv2 = nn.Conv1d(in_channels=32, 
                              out_channels=64, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=1, padding=7//2)
        
    def forward(self, x):
        # Lớp tích chập 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Lớp tích chập 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        return x