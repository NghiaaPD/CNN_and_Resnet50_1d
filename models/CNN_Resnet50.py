import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN import CNN1d
from Resnet import ResNet50

class CNN_Resnet50(nn.Module):
    def __init__(self, num_classes, num_channels=6):
        super(CNN_Resnet50, self).__init__()
        
        # Sử dụng CNN để trích xuất đặc trưng
        self.cnn = CNN1d(num_channels=num_channels)
        
        # ResNet50 với đầu vào là 64 kênh (output từ CNN)
        self.resnet = ResNet50(num_classes=num_classes, channels=64)
        
    def forward(self, x):
        # x có shape: [batch_size, 6, 7680]
        
        # Trích xuất đặc trưng bằng CNN
        features = self.cnn(x)
        # features có shape: [batch_size, 64, 7680]
        
        # Đưa qua ResNet50
        output = self.resnet(features)
        
        return output
