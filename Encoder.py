import torch
import torch.nn as nn
from utils import get_self_attention_padding_mask, generate_position_embedding
from MutiheadAttention import MutiHeadAttention
from PoswiseFeedForwardNet import PoswiseFeedForwardNet

class EncoderLayer(nn.Module):
    def __init__(self, paramters):
        super().__init__()
        self.paramters = paramters
        self.SelfAttention = MutiHeadAttention(paramters)
        self.FeedForward = PoswiseFeedForwardNet(paramters)
        self.LayerNorm = nn.LayerNorm(paramters["d_model"]).to(paramters["device"])
        
    def forward(self, EncoderInput, mask):
        # residual = EncoderInput
        EncoderOutput = self.SelfAttention(EncoderInput, EncoderInput, EncoderInput, mask)
        out = self.FeedForward(EncoderOutput)
        return out

class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.SrcWordEmbedding = nn.Embedding(parameters["src_vocab_size"], parameters["d_model"])
        self.PositionEmbedding = nn.Embedding.from_pretrained(generate_position_embedding(parameters["max_length"], parameters["d_model"], parameters["d_model"]), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(parameters) for _ in range(parameters["n_layers"])])
        self.dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, EncoderInput):
        seq_length = EncoderInput.shape[1]
        EncoderSelfMask = get_self_attention_padding_mask(EncoderInput).to(self.parameters["device"])
        ps_emb = self.PositionEmbedding(torch.arange(seq_length).to(self.parameters["device"]))
        word_emb = self.SrcWordEmbedding(EncoderInput)
        x = ps_emb + word_emb
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, EncoderSelfMask)
        return x