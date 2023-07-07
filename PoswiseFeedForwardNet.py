import torch
import torch.nn as nn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.parameter = p
        self.LinearLayer = nn.Sequential(
            nn.Linear(p["d_model"], p["d_ff"], bias=False),
            nn.ReLU(),
            nn.Linear(p["d_ff"], p["d_model"], bias=False)
        )
        self.LayerNorm = nn.LayerNorm(p["d_model"])
        
    def forward(self, x):
        residual = x
        output = self.LinearLayer(x)
        return self.LayerNorm(output + residual)