import torch
import torch.nn as nn
import numpy as np

class MutiHeadAttention(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.W_Q = nn.Linear(parameters["d_model"], parameters["d_model"], bias=False)
        self.W_K = nn.Linear(parameters["d_model"], parameters["d_model"], bias=False)
        self.W_V = nn.Linear(parameters["d_model"], parameters["d_model"], bias=False)
        self.softmax = nn.Softmax(-1)
        self.LayerNorm = nn.LayerNorm(parameters["d_model"]).to(parameters["device"])
        self.fc = nn.Linear(parameters["d_model"], parameters["d_model"], bias=False)
        
    def forward(self, Q, K, V, mask):
        batch_size, residual = Q.shape[0], Q
        Q = self.W_Q(Q).view(batch_size, -1, self.parameters["n_head"], self.parameters["d_k"]).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.parameters["n_head"], self.parameters["d_k"]).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.parameters["n_head"], self.parameters["d_v"]).transpose(1, 2)
        
        mask = mask.unsqueeze(1).repeat(1, self.parameters["n_head"], 1, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.parameters["d_k"])
        scores.masked_fill_(mask, -1e10)
        scores = self.softmax(scores)
        res = torch.matmul(scores, V)
        res = res.transpose(1, 2).reshape(batch_size, -1, self.parameters["d_model"])
        res = self.fc(res)
        return self.LayerNorm(res + residual)