import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CNN import CNN1d
from models.Resnet import ResNet50

class CNN_Resnet50_Features(nn.Module):
    """Feature extractor sử dụng cả CNN và ResNet50 để kết hợp với A2C"""
    
    def __init__(self, num_channels=6, num_classes=5):
        super(CNN_Resnet50_Features, self).__init__()
        
        self.cnn = CNN1d(num_channels=num_channels)
        
        self.resnet = ResNet50(num_classes=num_classes, channels=64)
        
        # Tách lớp linear cuối cùng để A2C có thể tối ưu hóa
        self.feature_dim = 512 * 4
        
    def forward(self, x):
        # Trích xuất đặc trưng với CNN
        x = self.cnn(x)
        
        # Xử lý qua ResNet (không sử dụng lớp cuối)
        # Truy cập các lớp bên trong ResNet
        x = self.resnet.relu(self.resnet.batch_norm1(self.resnet.conv1(x)))
        x = self.resnet.max_pool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        
        return x  # Trả về đặc trưng trước FC layer