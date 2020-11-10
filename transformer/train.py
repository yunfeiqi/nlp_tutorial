#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/09 11:14:42
@Author  :   sam.qi 
@Version :   1.0
@Desc    :   None
'''
from torch import optim
from transformer import Transformer
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

from dataset import TrainDataSet


vocab_size = 10
d_model = 20
d_q = 10
d_k = 10
d_v = 10
d_fnn = 50
n_head = 8
n_layer = 6


device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3,
             'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5  # enc_input max sequence length
tgt_len = 6  # dec_input(=dec_output) max sequence length


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

loader = Data.DataLoader(
    TrainDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

model = Transformer(vocab_size, d_model, d_q, d_k,
                    d_v, d_fnn, n_head, n_layer).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.8)

for epoch in range(10):
    for enc_inputs, dec_inputs, dec_outpus in loader:
        # enc_inputs : [batch_size,src_len]
        # dec_inputs : [batch_size,tgt_len]
        # dec_outpus : [batch_size,tgt_len]
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(
            device), dec_inputs.to(device), dec_outpus.to(device)

        #[batch * seq * vocab]
        outputs = model(enc_inputs, dec_inputs)
        # [N * Vocab]
        outputs = outputs.view(-1, outputs.size()[-1])
        loss = criterion(outputs, dec_outpus.view(-1))

        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Predict


def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """

    enc_outputs = model.encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


# Test
# [ batch * seq ]
enc_inputs, _, _ = next(iter(loader))
greedy_dec_input = greedy_decoder(model, enc_inputs[0].view(
    1, -1).to(device), start_symbol=tgt_vocab["S"])
predict = model(enc_inputs[0].view(1, -1).to(device), greedy_dec_input)
predict = predict.data.max(1, keepdim=True)[1]
print(enc_inputs[0], '->', [idx2word[n.item()] for n in predict.squeeze()])
