import torch.nn as nn
import torch
from torchvision import models

class ResnetCNNClass(nn.Module):
    def __init__(self, lstm_2048, lstm_layers, output_size, dropout, cnn, cnn_train, flow_bool, rpm_class, embedding_size, weight):
        super(ResnetCNNClass, self).__init__()
        # CNN
        self.resnet = getattr(models, cnn)(pretrained=True)
        self.cnn = nn.Sequential(*list(self.resnet.children())[:-1])
        self.cnn_out_features = 2048
        self.cnn_dropout = nn.Dropout(p=dropout)

        for param in self.cnn.parameters():
            param.requires_grad = cnn_train

        # RPM EMBEDDING, CONTINUOUS VERSION
        # self.rpm_embedding = nn.Embedding(rpm_class, self.embed_features)
        self.embed_features = embedding_size
        self.weight = weight

        # self.rpm_embedding = nn.Sequential(
        #     nn.Linear(1, self.embed_features),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_features, self.embed_features),
        # )
        
        self.temporalCNN = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )

        # LSTM
        # self.lstm = nn.LSTM(input_size=self.cnn_out_features, 2048=lstm_2048, 
        #                     num_layers=lstm_layers, batch_first=True, dropout=dropout)
        
        # FC LAYER
        self.flow_bool = flow_bool
        self.fc =nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(64, 50)
        )
        
    def forward(self, x, rpm):              
        batch_size, frames, C, H, W = x.shape
        x = x.view(batch_size * frames, C, H, W)

        # rpm_vec = self.rpm_embedding(rpm.unsqueeze(1))
        # rpm_vec = rpm_vec.unsqueeze(1).expand(-1, frames, -1)

        video_features = self.cnn(x) 
        video_features = video_features.view(batch_size, frames, -1) 

        # video_features = self.cnn_dropout(video_features)
        # concat = video_features + self.weight * rpm_vec
        concat = video_features
    
        # lstm_out, _ = self.lstm(concat)
        # lstm_last_out = lstm_out[:, -1, :]
        output = self.temporalCNN(concat.permute(0, 2, 1)).squeeze(-1)  # permute for shape

        if self.flow_bool:
            viscosity = lstm_last_out
        else:
            # viscosity = self.fc(lstm_last_out)
            viscosity = self.fc(output)
            
        return viscosity