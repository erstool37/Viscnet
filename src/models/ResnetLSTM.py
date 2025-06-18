import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class ResnetLSTM(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_layers, output_size, dropout, cnn, cnn_train, flow_bool):
        super(ResnetLSTM, self).__init__()
        self.resnet = getattr(models, cnn)(pretrained=True)
        self.cnn = nn.Sequential(*list(self.resnet.children())[:-1])
        self.cnn_out_features = 512

        self.flow_bool = flow_bool

        for param in self.cnn.parameters():
            param.requires_grad = cnn_train

        self.lstm = nn.LSTM(input_size=self.cnn_out_features, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout)
        
        self.fc =nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        """ x: (batch_size, sequence_length, channels, height, width)"""
        batch_size, frames, C, H, W = x.size()
        x = x.view(batch_size * frames, C, H, W)

        video_features = self.cnn(x) 
        video_features = video_features.view(batch_size, frames, -1) 

        lstm_out, _ = self.lstm(video_features)
        lstm_last_out = lstm_out[:, -1, :]
        lstm_last_out = lstm_last_out.view(batch_size, -1)
        
        if self.flow_bool:
            viscosity = lstm_last_out
        else:
            viscosity = self.fc(lstm_last_out)
        
        return viscosity