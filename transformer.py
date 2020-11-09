#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/09 14:27:22
@Author  :   sam.qi 
@Version :   1.0
@Desc    :   Transformer
'''

import math
import torch
import torch.nn as nn


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
        seq_len = data.size()[1]
        return self.dropout(data + self.pe[:seq_len, :])


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

        attn_q = torch.Tensor()

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

        attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)

        # [batch * seq * d_model]
        attn_output = self.fn(attn)
        return self.l_normal(attn_output + input_Q)


class ForwardNet(nn.Module):
    def __init__(self, d_model, d_fnn) -> None:
        super(ForwardNet, self).__init__()
        self.fnn1 = nn.Linear(d_model, d_fnn)
        self.fnn2 = nn.Linear(d_fnn, d_model)
        self.l_normal = nn.LayerNorm(d_model)

    def foward(self, data):
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


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_q, d_k, d_v, d_fnn,  n_layer, n_head) -> None:
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionEncoding(512, d_model=d_model)
        self.encoders = nn.ModuleList([EncoderLayer(d_model=d_model, n_head=n_head, d_q=d_q, d_k=d_k, d_v=d_v, d_fnn=d_fnn
                                                    ) for _ in range(n_layer)])

    def forward(self, data, attn_mask):
        '''
        data： [batch * seq]
        attn_mask: [batch * seq * seq]
        '''

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

    def forward(self, dec_input, enc_output, dec_mask, enc_mask):
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
            enc_output, enc_output, dec_output, dec_mask)

        output = self.fnn(attn_output)
        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_q, d_k, d_v, d_fnn, n_head, n_layer) -> None:
        super(Decoder, self).__init__()

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = PositionEncoding(512, d_model=d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, d_q=d_q, d_k=d_k,
                                                  d_v=d_v, d_fnn=d_fnn, n_head=n_head) for _ in range(n_layer)])

        self.fn = nn.Linear(d_model, vocab_size)

    def forward(self, data, attn_mask):
        emb = self.emb(data)
        output = self.pe(emb)
        for model in self.layers:
            output = model(output, attn_mask)

        result = nn.Softmax(dim=-1)(self.fn(output))
        return result


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_q, d_k, d_v, d_fnn,  n_layer, n_head) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, d_q,
                               d_k, d_v, d_fnn,  n_layer, n_head)
        self.decoder = Decoder(vocab_size, d_model, d_q,
                               d_k, d_v, d_fnn, n_head, n_layer)

    def forward(self, enc_input, dec_input, enc_mask, dec_mask):
        '''
        enc_input ： [batch * enc_seq]
        dec_input ： [batch * dec_seq]
        enc_mask:[batch * enc_seq * enc_seq]
        dec_mask:[batch * dec_seq * dec_seq]
        '''

        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(dec_input, enc_output, enc_mask, dec_mask)
        return dec_output
