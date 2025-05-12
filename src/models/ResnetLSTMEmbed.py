import torch.nn as nn
import torch
from torchvision import models

class ResnetLSTMEmbed(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_layers, output_size, dropout, cnn, cnn_train, flow_bool, rpm_class, embedding_size):
        super(ResnetLSTMEmbed, self).__init__()
        self.resnet = getattr(models, cnn)(pretrained=True)
        self.cnn = nn.Sequential(*list(self.resnet.children())[:-1])
        self.cnn_out_features = 512
        self.flow_bool = flow_bool
        self.embed_features = embedding_size

        for param in self.cnn.parameters():
            param.requires_grad = cnn_train

        self.rpm_embedding = nn.Embedding(rpm_class, self.embed_features)

        self.lstm = nn.LSTM(input_size=self.cnn_out_features + self.embed_features, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout)
        self.fc =nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )
        
    def forward(self, x, rpm_idx):
        """ 
        x: (B, L, C, H, W)
        rpm_idx : 
        """
        batch_size, frames, C, H, W = x.shape
        x = x.view(batch_size * frames, C, H, W)

        rpm_vec = self.rpm_embedding(rpm_idx.long())
        rpm_vec = rpm_vec.unsqueeze(1).expand(-1, frames, -1)
        
        video_features = self.cnn(x) 
        video_features = video_features.view(batch_size, frames, -1) 

        concat = torch.cat((video_features, rpm_vec), dim=2)

        lstm_out, _ = self.lstm(concat)
        lstm_last_out = lstm_out[:, -1, :]

        if self.flow_bool:
            viscosity = lstm_last_out
        else:
            viscosity = self.fc(lstm_last_out)
        
        return viscosity