from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


__all__ = ['ffn']




class FeedForwardNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 10)        
        self.init()
        
    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


    def track_running_stats(self, state):
        dummy = state


    def forward(self, x):
        # x = x[:,0,:,:]
        # x = F.max_pool2d(x, 4)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x


def ffn(**kwargs):
    model = FeedForwardNet(**kwargs)
    return model
