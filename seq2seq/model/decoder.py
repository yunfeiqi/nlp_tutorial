#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/1 1:55 下午
# @Author  : Sam
# @Desc    :


import torch.nn as nn

from model.attention import Attention


class RNNDecoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
        self.isatt = isatt
        self.attn = Attention()

        self.input_dim = emb_dim
        self.rnn = nn.GRU(self.input_dim, self.hid_dim,
                          self.n_layers, dropout=dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_output, source_mask):
        # input: [batch, vocab_size]
        # hidden: [batch ,layer * direction,hid_dim]

        input = input.unsqueeze(1)
        # [batch , 1, emb_dim]
        embeded = self.dropout(self.embedding(input))

        # output: [batch ,1, hid_dim ]
        # hidden :[n_layer,batch size,hid_dim]

        if self.isatt:
            # 如果采用Attention
            attn = self.attn(encoder_output, hidden, source_mask)
            hidden = hidden + attn
        output, hidden = self.rnn(embeded, hidden)
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        return prediction, hidden
