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

# def get_self_attention_padding_mask(seq_q, seq_k):
#     '''
#     seq_q: [batch_size, seq_len]
#     seq_k: [batch_size, seq_len]
#     seq_len could be src_len or it could be tgt_len
#     seq_len in seq_q and seq_len in seq_k maybe not equal
#     '''
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#     # eq(zero) is PAD token
#     # [batch_size, 1, len_k], False is masked
#     pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
#     # [batch_size, len_q, len_k]
#     return pad_attn_mask.expand(batch_size, len_q, len_k)


# def get_subsequence_mask(seq):
#     '''
#     seq: [batch_size, tgt_len]
#     '''
#     attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
#     # Upper triangular matrix
#     subsequence_mask = np.triu(np.ones(attn_shape), k=1)
#     subsequence_mask = torch.from_numpy(subsequence_mask).byte()
#     return subsequence_mask  # [batch_size, tgt_len, tgt_len]

def generate_position_embedding(max_length, embedding_size, d_model):
    ps = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(embedding_size)] for pos in range(max_length)
    ])
    ps[:, 0::2] = np.sin(ps[:, 0::2])
    ps[:, 1::2] = np.cos(ps[:, 0::2])
    return torch.tensor(ps, dtype=torch.float32)