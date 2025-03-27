import torch
import torch.nn as nn
import torch.nn.functional as F
from .CNN import CNN1d
from .Resnet import ResNet50

class CNN_Resnet50(nn.Module):
    def __init__(self, num_classes, num_channels=6):
        super(CNN_Resnet50, self).__init__()
        
        self.cnn = CNN1d(num_channels=num_channels)

        self.resnet = ResNet50(num_classes=num_classes, channels=64)
        
    def forward(self, x):

        features = self.cnn(x)

        output = self.resnet(features)
        
        return output
