import torch.nn as nn
import torch
from torchvision import models

class Resnet(nn.Module):
    def __init__(self, output_size):
        super(Resnet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.resnet.children())[:-1])
        self.cnn_out_features = 512

        for param in self.cnn.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(self.cnn_out_features, output_size)
    
    def forward(self, x):
        """ x: (batch_size, sequence_length, channels, height, width)"""
        batch_size, frames, C, H, W = x.size()
        x = x.view(batch_size * frames, C, H, W)

        video_features = self.cnn(x) 
        video_features = video_features.view(batch_size, frames, -1) 
        video_features = video_features.mean(dim=1)
        
        viscosity = self.fc(video_features)  # (batch_size, output_size)
        return viscosity