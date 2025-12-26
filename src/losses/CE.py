import torch
import torch.nn as nn

class CE(nn.Module):
    def __init__(self, unnormalizer, path, smth_label):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, hotvector):
        loss = self.criterion(outputs, hotvector)
        return loss