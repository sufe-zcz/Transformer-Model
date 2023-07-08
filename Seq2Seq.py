import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransform
from torchtext.transforms import AddToken
from torchtext.transforms import Truncate
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import os
from transformer import Transformer
from Translator import Translator

min_freq = 5
batch_size = 32
num_samples = 20

data = pd.read_csv("translation_data.csv")
data["Chinese"] = data["Chinese"].apply(lambda x:x[:-1])
data["English"] = data["English"].apply(lambda x:x[:-1])
data["Englist_lower"] = data["English"].apply(lambda x:x.lower())

def tokenizer_english(text):
    return text.split(" ")

def tokenizer_chinese(text):
    return list(text)
    
def yield_data(data, col, tokenizer):
    for idx, items in data[[col]].iterrows():
        yield [item for item in tokenizer(items[0])]

source_vocab = build_vocab_from_iterator(
    yield_data(data, "Englist_lower", tokenizer_english),
    min_freq=min_freq,
    specials=["<PAD>", "<UNK>", "<BEGIN>", "<END>"]
    )

target_vocab = build_vocab_from_iterator(
    yield_data(data, "Chinese", tokenizer_chinese),
    min_freq=min_freq,
    specials=["<PAD>", "<UNK>", "<BEGIN>", "<END>"]
    )

source_vocab.set_default_index(source_vocab["<UNK>"])
target_vocab.set_default_index(target_vocab["<UNK>"])

source_vocab_transformer = VocabTransform(source_vocab)
target_vocab_transformer = VocabTransform(target_vocab)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data[["Englist_lower", "Chinese"]]
        self.data.columns = ["source", "target"]
        
    def __getitem__(self, idx):
        src = tokenizer_english(self.data["source"][idx])
        tgt1 = tokenizer_chinese(self.data["target"][idx])
        tgt2 = tokenizer_chinese(self.data["target"][idx])
        src = source_vocab_transformer(src)
        tgt1 = [target_vocab["<BEGIN>"]] + target_vocab_transformer(tgt1)
        tgt2 = target_vocab_transformer(tgt2) + [target_vocab["<END>"]]
        return src, tgt1, tgt2
    
    def __len__(self):
        return len(self.data)

def my_collate(batch):
    rec_list, target_list, target_list_ = [], [], []
    for record in batch:
        if len(record[0]) > 100:
            record[0] = record[0][:100]
        if len(record[1]) > 100:
            record[1] = record[1][:100]
        if len(record[2]) > 100:
            record[2] = record[2][:100]
        rec_list.append(torch.tensor(record[0]))
        target_list.append(torch.tensor(record[1]))
        target_list_.append(torch.tensor(record[2]))
    EncoderInput = pad_sequence(rec_list, padding_value=0,batch_first=True)
    DecoderInput1 = pad_sequence(target_list, padding_value=0,batch_first=True)
    DecoderInput2 = pad_sequence(target_list_, padding_value=0,batch_first=True)
    return EncoderInput,  DecoderInput1, DecoderInput2

dataset = MyDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate)

device = "cuda"
parameter = {
    "d_model" : 512,
    "d_ff" : 512 * 4,
    "d_k" : 64,
    "d_v" : 64,
    "n_layers": 6,
    "n_head" : 8,
    "max_length" : 100,
    "src_vocab_size":len(source_vocab),
    "tgt_vocab_size":len(target_vocab),
    "device" : device,
    "output_dim": len(target_vocab)
}

train = False
save = False

model = Transformer(parameter).to(device)
save_path = "./save/model.pth"
if os.path.exists(save_path):
    print(f"Checkpoint exists. Loading from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))

loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
print(f"Training on {device}")
if train:
    for epoch in range(20):
        l = 0.0
        for idx, (encoder_input, decoder_input, decoder_output) in enumerate(dataloader):
            encoder_input, decoder_input, decoder_output = encoder_input.to(device), decoder_input.to(device), decoder_output.to(device)
            output = model(encoder_input, decoder_input)
            output = output.view(-1, output.shape[-1])
            optimizer.zero_grad()
            # print(output.shape)
            loss = loss_function(output, decoder_output.view(-1))
            loss.backward()
            optimizer.step()
            l += loss.detach().item()
            if idx % 200 == 0 and idx:
                print(f"loss : {loss.item():.2f}")
        if epoch % 10 == 0 and epoch != 0 and save:
            torch.save(model.state_dict(), f'save/model_EPOCH_{epoch}.pth')
        print(f"EPOCH : {epoch}\tLOSS : {l:.2f}")

# if save:
#     torch.save(model.state_dict(), f'save/model.pth')


tranlator = Translator(model, source_vocab, target_vocab, tokenizer_english, device)
text = "you are my son"

tranlator.render(text)


