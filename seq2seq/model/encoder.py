#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/1 1:55 下午
# @Author  : Sam
# @Desc    : RNNEncoder

import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout=0.02):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # 双向RNN
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers,
                          dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input: batch * sequence_len

        # emb: batch * seq_len * emb_dim
        emb = self.embedding(input)

        # outputs: batch * sequence *  (hid_dim * bidir)
        # hidden:  (n_layer * direction) * batch * hid_dim
        outputs, hidden = self.rnn(self.dropout(emb))

        return outputs, hidden
