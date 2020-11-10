#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/09 14:27:22
@Author  :   sam.qi 
@Version :   1.0
@Desc    :   Transformer
'''

import math
from numpy.core.numeric import base_repr
import torch
import torch.nn as nn
import numpy as np


class Base(nn.Module):
    def __init__(self) -> None:
        super(Base, self).__init__()

    def get_attn_pad_mask(self, seq_q, seq_k):
        '''
        seq_q: [batch * seq_q]
        seq_k: [batch * seq_k]

        Atten_Q: [batch * seq_q * dim]
        Atten_K: [batch * seq_k * dim]
        Attn: [ batch * seq_q * seq_k]
        '''

        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        #[ batch * 1 * seq_k ]
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)

    def get_attn_subsequence_mask(self, seq):
        '''
        seq: [batch_size, tgt_len]
        '''
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        # Upper triangular matrix
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class PositionEncoding(nn.Module):
    def __init__(self, max_seq_size, d_model, dropout=0.1) -> None:
        super(PositionEncoding, self).__init__()
        self.max_seq_size = max_seq_size
        self.d_model = d_model

        # [max_seq * d_model]
        pe = torch.zeros(self.max_seq_size, self.d_model, dtype=torch.float)

        # [max_seq * 1]
        position = torch.arange(max_seq_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [max_seq * 1 * d_model]
        self.pe = pe.unsqueeze(0).transpose(0, 1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        '''
        data: [ batch * seq * d_model]
        '''
        data = data.transpose(0, 1)
        seq_len = data.size()[0]
        output = self.dropout(data + self.pe[:seq_len, :])
        return output.transpose(0, 1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, attn_q, attn_k, attn_v, attn_mask):
        '''
        attn_q : [batch * head * seq * d_k]
        attn_k : [batch * head * seq * d_k]
        attn_v : [batch * head * seq * d_v]
        attn_mask: [batch * head * seq * seq]
        '''

        # [batch * head *  seq * seq]
        weight = attn_q.matmul(attn_k.transpose(-1, -2))
        weight.masked_fill_(attn_mask, -1e9)

        # [batch * head *  seq * seq]
        score = nn.Softmax(dim=-1)(weight)
        # [batch * head *  seq * d_v ]
        output = score.matmul(attn_v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_q, d_k, d_v) -> None:
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_q*n_head)
        self.w_k = nn.Linear(d_model, d_k*n_head)
        self.w_v = nn.Linear(d_model, d_v*n_head)

        self.fn = nn.Linear(d_q * n_head, d_model)
        self.l_normal = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        data: [batch * seq * d_model]
        attn_mask : [batch * seq * seq]
        '''

        batch = input_Q.size()[0]

        # Transfor input data to Q K and V

        # [ batch * head * seq * d_k ]
        Q = self.w_q(input_Q).view(batch, self.n_head, -1, self.d_k)
        K = self.w_k(input_K).view(batch, self.n_head, -1, self.d_k)
        # [ batch * head * seq * d_v ]
        V = self.w_v(input_V).view(batch, self.n_head, -1, self.d_v)

        # Transformer atten_mask to [batch * head * seq * seq]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # [ batch * n_head * seq * d_v ]
        attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)

        # # [ batch * seq * d_v_n_head ]
        attn = attn.transpose(1, 2).reshape(batch, -1, self.n_head * self.d_v)

        # [batch * seq * d_model]
        attn_output = self.fn(attn)
        return self.l_normal(attn_output + input_Q)


class ForwardNet(nn.Module):
    def __init__(self, d_model, d_fnn) -> None:
        super(ForwardNet, self).__init__()
        self.fnn1 = nn.Linear(d_model, d_fnn)
        self.fnn2 = nn.Linear(d_fnn, d_model)
        self.l_normal = nn.LayerNorm(d_model)

    def forward(self, data):
        '''
        data :[ batch * seql * d_model]
        '''

        output_fnn1 = self.fnn1(data)
        output_fnn2 = self.fnn2(output_fnn1)

        return self.l_normal(data + output_fnn2)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_q, d_k, d_v, d_fnn) -> None:
        super(EncoderLayer, self).__init__()
        self.multi_attention = MultiHeadAttention(
            d_model=d_model, n_head=n_head, d_q=d_q, d_k=d_k, d_v=d_v)
        self.fnn = ForwardNet(d_model, d_fnn)

    def forward(self, data, mask):
        '''
            data: [batch * seq * d_model]
            mask: [batch * seq * seq]
        '''

        attn_output = self.multi_attention(data, data, data, mask)
        output = self.fnn(attn_output)
        return output


class Encoder(Base):
    def __init__(self, vocab_size, d_model, d_q, d_k, d_v, d_fnn,  n_layer, n_head) -> None:
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionEncoding(512, d_model=d_model)
        self.encoders = nn.ModuleList([EncoderLayer(d_model=d_model, n_head=n_head, d_q=d_q, d_k=d_k, d_v=d_v, d_fnn=d_fnn
                                                    ) for _ in range(n_layer)])

    def forward(self, data):
        '''
        data： [batch * seq]
        '''

        # mask for input
        # [batch * seq * seq]
        attn_mask = self.get_attn_pad_mask(data, data)

        # [batch * seq * d_model]
        output_emb = self.emb(data)
        # [batch * seq * d_model]
        output = self.pos_emb(output_emb)
        for m in self.encoders:
            output = m(output, attn_mask)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, d_fnn, n_head) -> None:
        super(DecoderLayer, self).__init__()
        self.src_multi_attention = MultiHeadAttention(
            d_model=d_model, n_head=n_head, d_q=d_q, d_k=d_k, d_v=d_v)

        self.enc_multi_attention = MultiHeadAttention(
            d_model=d_model, n_head=n_head, d_q=d_q, d_k=d_k, d_v=d_v)
        self.fnn = ForwardNet(d_model, d_fnn)

    def forward(self, dec_input, enc_output, dec_mask, dec_enc_mask):
        '''
        dec_input : [batch * dec_seq * d_model]
        enc_output: [batch * end_seq * d_model]
        dec_mask : [batch * dec_seq * dec_seq]
        enc_mask: [batch * end_seq * end_seq]
        '''

        # [batch * dec_sql * d_model]
        dec_output = self.src_multi_attention(
            dec_input, dec_input, dec_input, dec_mask)
        attn_output = self.src_multi_attention(
            dec_output, enc_output, enc_output, dec_enc_mask)

        output = self.fnn(attn_output)
        return output


class Decoder(Base):
    def __init__(self, vocab_size, d_model, d_q, d_k, d_v, d_fnn, n_head, n_layer) -> None:
        super(Decoder, self).__init__()

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = PositionEncoding(512, d_model=d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, d_q=d_q, d_k=d_k,
                                                  d_v=d_v, d_fnn=d_fnn, n_head=n_head) for _ in range(n_layer)])

    def forward(self, dec_input, enc_input, enc_output):
        '''
        dec_input : [ batch * seq]
        enc_input : [ batch * seq_enc]
        enc_output : [ batch * seql * d_model]
        '''

        # pad mask for decoder
        dec_src_mask = self.get_attn_pad_mask(dec_input, dec_input)
        # subsqequence mask for decoder
        dec_sub_mask = self.get_attn_subsequence_mask(dec_input)
        # decoder mask
        dec_mask = torch.gt(dec_src_mask + dec_sub_mask, 0)
        # decoder - encoder attention mask
        dec_enc_attn_mask = self.get_attn_pad_mask(dec_input, enc_input)

        emb = self.emb(dec_input)
        output = self.pe(emb)
        for model in self.layers:
            output = model(output, enc_output, dec_mask, dec_enc_attn_mask)

        return output


class Transformer(Base):
    def __init__(self, vocab_size, d_model, d_q, d_k, d_v, d_fnn,  n_layer, n_head) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, d_q,
                               d_k, d_v, d_fnn,  n_layer, n_head)
        self.decoder = Decoder(vocab_size, d_model, d_q,
                               d_k, d_v, d_fnn, n_head, n_layer)

        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, enc_input, dec_input):
        '''
        enc_input ： [batch * enc_seq]
        dec_input ： [batch * dec_seq]
        enc_mask:[batch * enc_seq * enc_seq]
        dec_mask:[batch * dec_seq * dec_seq]

        output:  [batch * dec_seq * vocab]
        '''

        enc_output = self.encoder(enc_input)

        # [batch * seq * d_model]
        dec_output = self.decoder(dec_input, enc_input, enc_output)

        # [batch * seq * vocab]
        dec_logist = self.projection(dec_output)

        return dec_logist
