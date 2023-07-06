import torch
import torch.nn as nn
from utils import get_self_attention_padding_mask, generate_position_embedding
from MutiheadAttention import MutiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, paramters):
        super().__init__()
        self.paramters = paramters
        self.SelfAttention = MutiHeadAttention(paramters)
        self.fc = nn.Sequential(
            nn.Linear(paramters["d_model"], paramters["d_ff"]),
            nn.ReLU(),
            nn.Linear(paramters["d_ff"], paramters["d_model"])
        )
        self.LayerNorm = nn.LayerNorm(paramters["d_model"]).to(paramters["device"])
        
    def forward(self, EncoderInput, mask):
        # residual = EncoderInput
        EncoderOutput = self.SelfAttention(EncoderInput, EncoderInput, EncoderInput, mask)
        residual = EncoderOutput
        out = self.fc(EncoderOutput)
        return self.LayerNorm(out + residual)

class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.SrcWordEmbedding = nn.Embedding(parameters["src_vocab_size"], parameters["d_model"])
        self.PositionEmbedding = nn.Embedding.from_pretrained(generate_position_embedding(parameters["max_length"], parameters["d_model"], parameters["d_model"]), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(parameters) for _ in range(parameters["n_layers"])])
        
    def forward(self, EncoderInput):
        seq_length = EncoderInput.shape[1]
        EncoderSelfMask = get_self_attention_padding_mask(EncoderInput).to(self.parameters["device"])
        ps_emb = self.PositionEmbedding(torch.arange(seq_length).to(self.parameters["device"]))
        word_emb = self.SrcWordEmbedding(EncoderInput)
        x = ps_emb + word_emb
        for layer in self.layers:
            x = layer(x, EncoderSelfMask)
        return x