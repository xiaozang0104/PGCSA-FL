# model_MNIST2.py

import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入 1×28×28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)   # -> 32×28×28
        self.pool  = nn.MaxPool2d(2, 2)                          # -> 32×14×14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # -> 64×14×14
        # 再次下采样
        # self.pool -> 64×7×7
        self.pool2 = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)  # MNIST 10 类

    def forward(self, x):
        # x: (B,1,28,28)
        x = F.relu(self.conv1(x))
        x = self.pool(x)               # (B,32,14,14)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)              # (B,64,7,7)
        x = x.view(x.size(0), -1)      # 展平 -> (B,64*7*7)
        x = F.relu(self.fc1(x))        # (B,128)
        x = self.fc2(x)                # (B,10)
        return x
