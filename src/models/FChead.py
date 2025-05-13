import torch
import torch.nn as nn
from torchvision import models

class FChead(nn.Module):
    def __init__(self, lstm_hidden_size, output_size):
        super(FChead, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.05),

            # nn.Linear(256, 128),
            # nn.LayerNorm(128),
            # nn.ReLU(),
            # nn.Dropout(0.05),

            # nn.Linear(128, 64),
            # nn.ReLU(),

            nn.Linear(256, 32),
            nn.ReLU(),

            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.fc(x)
