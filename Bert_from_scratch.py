import torch
import torch.nn as nn
import numpy as np


def get_params(model):
    return sum(l.numel() for l in model.parameters())

def get_self_attention_pad_mask(seq):
    batch_size, seq_len = seq.shape
    mask = seq.eq(0).unsqueeze(1)
    mask = mask.expand(batch_size, seq_len, seq_len)
    return mask

class FC(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.LeakyReLU(),
            nn.Linear(4 * embedding_size, embedding_size)
        )
        self.LayerNorm = nn.LayerNorm(embedding_size)
    
    def forward(self, x):
        residual = x
        return self.LayerNorm(self.model(x) + residual)

class FC_attention(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.LayerNorm = nn.LayerNorm(embedding_size)
    
    def forward(self, x):
        residual = x
        return self.LayerNorm(self.fc(x) + residual)
    

class SelfAttention(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.W_Q = nn.Linear(embedding_size, head_size)
        self.W_K = nn.Linear(embedding_size, head_size)
        self.W_V = nn.Linear(embedding_size, head_size)
        self.softmax = nn.Softmax(-1)
        self.LayerNorm = nn.LayerNorm(head_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_Q, input_K, input_V, mask=None):
        Q = self.W_Q(input_Q) # [batch_size, seq_len, embedding_size]
        K = self.W_K(input_K)
        V = self.W_V(input_V)
        
        scores = torch.bmm(Q, K.transpose(-1, -2)) / np.sqrt(self.head_size)
        if mask is not None:
            scores.masked_fill_(mask, 1e-10)
        scores = self.dropout(self.softmax(scores))
        return torch.bmm(scores, V)
    
class MutiHeadAttention(nn.Module):
    def __init__(self, embedding_size, n_head, head_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_head = n_head
        self.head_size = head_size
        assert embedding_size // n_head == head_size
        self.model_list = nn.ModuleList([SelfAttention(embedding_size, head_size) for _ in range(n_head)])
        self.fc = FC_attention(embedding_size)
    
    def forward(self, input_Q, input_K, input_V, mask=None):
        value_output = [m(input_Q, input_K, input_V, mask) for m in self.model_list]
        outputs = torch.cat(value_output, dim=-1)
        outputs = self.fc(outputs)
        return outputs
            
class BertLayer(nn.Module):
    def __init__(self, embedding_size, n_head, head_size):
        super().__init__()
        self.Attention = MutiHeadAttention(embedding_size, n_head, head_size)
        self.fc = FC(embedding_size)
        
    def forward(self, x, mask=None):
        x = self.Attention(x, x, x, mask)
        output = self.fc(x)
        return output
    
class Bert(nn.Module):
    def __init__(self, num_layers, embedding_size, n_head, head_size, device):
        super().__init__()
        self.WE = nn.Embedding(10, 512)
        self.PE = nn.Embedding(10, 512)
        self.num_layers = num_layers
        self.device = device
        self.embedding_size = embedding_size
        self.n_head = n_head
        self.head_size = head_size
        self.dropout = nn.Dropout(0.1)
        self.Layers = nn.ModuleList([BertLayer(embedding_size, n_head, head_size) for _ in range(num_layers)])
        
    def forward(self, x):
        mask = get_self_attention_pad_mask(x).to(self.device)
        we = self.WE(x)
        pe = self.PE(x)
        x_emb = self.dropout(we + pe)
        for layer in self.Layers:
            x_emb = layer(x_emb, mask)
        return x_emb



if __name__ == '__main__':
    batch_size = 3
    max_seq_len = 5
    embedding_size = 512
    num_head = 8
    head_size = 64
    num_layers = 12
    device = "mps"
    a = torch.LongTensor(
        [[1, 2, 3, 1, 2, 0],
        [3, 5, 2, 1, 1, 1],
        [1, 2, 0, 0, 0, 0]]
    ).to(device)
    model = Bert(num_layers, embedding_size, num_head, head_size, device).to(device)
    print(model(a).shape)
    print(get_params(model))
    