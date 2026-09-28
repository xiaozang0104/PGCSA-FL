# model_cifar10.py

from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):      # CIFAR-10 一共 10 类
        super().__init__()
        # (3,32,32) -> (32,30,30)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)       # -> (32,15,15)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)       # -> (64,6,6)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # conv3 之后输出是 (64,4,4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # 直接假设输入已经是 (B,3,32,32)
        x = F.relu(self.conv1(x))   # (B,32,30,30)
        x = self.pool1(x)           # (B,32,15,15)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.conv2(x))   # (B,64,13,13)
        x = self.pool2(x)           # (B,64,6,6)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.conv3(x))   # (B,64,4,4)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
