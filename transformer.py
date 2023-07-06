import torch
import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.Encoder = Encoder(p)
        self.Decoder = Decoder(p)
        self.fc = nn.Linear(p["d_model"], p["output_dim"])
        
    def forward(self, EncoderInput, DecoderInput):
        EncoderOutput = self.Encoder(EncoderInput)
        DecoderOutput = self.Decoder(EncoderInput, EncoderOutput, DecoderInput)
        return self.fc(DecoderOutput)
        