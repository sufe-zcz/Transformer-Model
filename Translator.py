import torch
import torch.nn as nn
import pandas as pd
import os
from transformer import Transformer
from torchtext.transforms import VocabTransform
from torch.distributions import Categorical

class Translator(object):
    def __init__(self, model, src_vocab, tgt_vocab, tokenizer, device):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.source_vocab_transformer = VocabTransform(src_vocab)
        self.target_vocab_transformer = VocabTransform(tgt_vocab)
        self.tokenizer = tokenizer

    def load_model(self, path):
        if os.path.exists(path):
            print(f"Loading model from {path}")
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print("Path does not exist !!")

    def greedy_decoder(self, enc_input, start_symbol):
        enc_outputs = self.model.Encoder(enc_input)
        dec_input = torch.zeros(1, 0).type_as(enc_input.data)
        terminal = False
        next_symbol = start_symbol
        translation = []
        while not terminal:
            dec_input = torch.cat([dec_input.detach(), torch.tensor(
                [[next_symbol]], dtype=enc_input.dtype).to(self.device)], -1)
            # EncoderInput, EncoderOutput, DecoderInput
            dec_outputs = self.model.Decoder(enc_input, enc_outputs, dec_input)
            projected = self.model.fc(dec_outputs)
            # prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            
            prob = nn.Softmax(dim=-1)(projected.squeeze(0)[-1, :])
            category = Categorical(prob)
            
            # next_word = prob.data[-1]
            # next_symbol = next_word
            next_word = category.sample()
            next_symbol = next_word
            if next_symbol == self.tgt_vocab["<END>"]:
                terminal = True
            translation.append(next_symbol)
            # print(next_word)
        return dec_input

    def render(self, text):
        print(text, "\t->\t", end="")
        text = text.lower()
        text = self.source_vocab_transformer(self.tokenizer(text))[:100]
        text = torch.LongTensor(text).unsqueeze(0).to(self.device)
        output = self.greedy_decoder(text, start_symbol=self.tgt_vocab["<BEGIN>"])
        translation = "".join([self.tgt_vocab.get_itos()[item.item()] for item in output.squeeze(0)][1:])
        print(translation)
        return translation

