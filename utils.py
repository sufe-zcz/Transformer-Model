import torch
import torch.nn as nn
import numpy as np

def get_self_attention_padding_mask(seq_source, seq_target=None):
    if seq_target is None:
        batch_size, seq_length = seq_source.shape
        mask = seq_source.eq(0).unsqueeze(1)
        mask = mask.expand(batch_size, seq_length, seq_length)
    else:
        batch_size, source_length = seq_source.shape
        batch_size, target_length = seq_target.shape
        mask = seq_source.eq(0).unsqueeze(1)
        mask = mask.expand(batch_size, target_length, source_length)
    return mask
        
def get_subsequence_mask(seq):
    mask_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]
    mask = np.triu(np.ones(shape=mask_shape), k=1)
    return torch.from_numpy(mask).byte()


def generate_position_embedding(max_length, embedding_size, d_model):
    ps = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(embedding_size)] for pos in range(max_length)
    ])
    ps[:, 0::2] = np.sin(ps[:, 0::2])
    ps[:, 1::2] = np.cos(ps[:, 0::2])
    return torch.tensor(ps, dtype=torch.float32)