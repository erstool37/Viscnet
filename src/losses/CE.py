import torch
import torch.nn as nn

class CE(nn.Module):
    def __init__(self, unnormalizer, path):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, hotvector):
        loss = self.criterion(logits, hotvector)
        return loss