import torch
import torch.nn as nn
import numpy as np
from MutiheadAttention import MutiHeadAttention
from utils import *

class DecoderLayer(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.SelfAttention = MutiHeadAttention(parameters)
        self.EncoderDecoderAttention = MutiHeadAttention(parameters)
        self.fc = nn.Sequential(
            nn.Linear(parameters["d_model"], parameters["d_ff"]),
            nn.ReLU(),
            nn.Linear(parameters["d_ff"], parameters["d_model"])
        )
        self.LayerNorm = nn.LayerNorm(parameters["d_model"])
        
    def forward(self, 
                DecoderInput,
                EncoderOutput,
                DecoderSelfMask,
                DecoderEncoderPadMask
                ):
        DecoderOutput = self.SelfAttention(DecoderInput, DecoderInput, DecoderInput, DecoderSelfMask)
        DecoderOutput = self.EncoderDecoderAttention(
            Q=DecoderOutput,
            K=EncoderOutput,
            V=EncoderOutput,
            mask=DecoderEncoderPadMask
        )
        residual = DecoderOutput
        out = self.fc(DecoderOutput)
        return self.LayerNorm(out + residual)
    
class Decoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.TgtWordEmbedding = nn.Embedding(
            parameters["tgt_vocab_size"], parameters["d_model"]
        )
        self.PositionEmbedding = nn.Embedding.from_pretrained(
            generate_position_embedding(
                parameters["max_length"],
                parameters["d_model"],
                parameters["d_model"]
            ),
            freeze=True
        )
        self.layers = nn.ModuleList([DecoderLayer(parameters) for _ in range(parameters["n_layers"])])
    
    def forward(self, EncoderInput, EncoderOutput, DecoderInput):
        seq_length = DecoderInput.shape[1]
        WordEmb = self.TgtWordEmbedding(DecoderInput)
        PosEmb = self.PositionEmbedding(torch.arange(seq_length).to(self.parameters["device"]))
        x = WordEmb + PosEmb
        DecoderSelfPadMask = get_self_attention_padding_mask(DecoderInput).to(self.parameters["device"])
        DecoderSelfSeqMask = get_subsequence_mask(DecoderInput).to(self.parameters["device"])
        DecoderSelfMask = torch.gt(DecoderSelfPadMask + DecoderSelfSeqMask, 0).to(self.parameters["device"])
        
        DecoderEncoderMask = get_self_attention_padding_mask(
            seq_source=EncoderInput,
            seq_target=DecoderInput
        )
        
        for layer in self.layers:
            x = layer(x, EncoderOutput, DecoderSelfMask, DecoderEncoderMask)
        return x
