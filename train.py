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
import math
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
                    d_v, d_fnn, n_head, n_layer).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters, lr=1e-3, momentum=0.8)

for epoch in range(30):
    for enc_inputs, dec_inputs, dec_outpus in loader:
        # enc_inputs : [batch_size,src_len]
        # dec_inputs : [batch_size,tgt_len]
        # dec_outpus : [batch_size,tgt_len]
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(
        ), dec_inputs.cuda(), dec_outpus.cuda()

        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(
            enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outpus.view(-1))

        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
