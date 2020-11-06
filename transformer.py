#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/06 13:11:49
@Author  :   sam.qi 
@Version :   1.0
@Desc    :   该文件主要是Transformer模型组件包括
        PositionalEncoding: 位置编码
        ScaledDotProductAttention： 基于内积的Attention
        MultiHeadAttention： 多头注意力
        FeedForward: 前馈网络
        Encoder：Encoder 模型
        Decoder: Decoder 模型
        Transformer: Transformer 模型
        
'''

from typing import Match
from torch.nn.modules import dropout

import torch
import math
import torch.nn as nn
import numpy as np
from torch.nn.modules.activation import MultiheadAttention, ReLU
from torch.nn.modules.linear import Linear


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, dim_size, dropout=0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.dim_size = dim_size

        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding
        pe = torch.zeros(max_seq_len, dim_size)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, dim_size, 2).float() * (-math.log(10000)/dim_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        '''
        @Desc    :   x : [seql_len,batch_size,d_model]
        @Time    :   2020/11/06 13:31:11
        @Author  :   sam.qi 
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        '''
        @Desc    :   
            Q: [batch,n_head,len_q,d_k]
            K: [batch,n_head,len_k,d_k]
            V: [batch,n_head,len_v,d_k]
            attn_mask = [batch,n_heads,seq_len,seq_len]
        @Time    :   2020/11/06 13:42:49
        @Author  :   sam.qi 
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # 此时不可以马上进行 softmax 必须将部分信息屏蔽
        scores.masked_fill_(attn_mask, -1e9)

        weight = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(weight, V)
        return context, weight


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads) -> None:
        super(MultiheadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.n_head = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

    def forawrd(self, input_Q, input_K, input_V, attn_mask):
        '''
        @Desc    :   
            input_Q: [batch,seq_len,d_model]
            input_W: [batch,seq_len,d_model]
            input_V: [batch,seq_len,d_model]
            atten_mask : [batch,seq_len,seq_len]
        @Time    :   2020/11/06 14:06:40
        @Author  :   sam.qi 
        '''

        residual = input_Q
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, self.n_head, -1,
                                   self.d_k)  # [Batch,Head,SeqLen,Dimension]
        K = self.W_Q(input_K).view(batch_size, self.n_head, -1,
                                   self.d_k)  # [Batch,Head,SeqLen,Dimension]
        V = self.W_Q(input_V).view(batch_size, self.n_head, -1,
                                   self.d_k)  # [Batch,Head,SeqLen,Dimension]

        # [batch,n_head,seq_len,seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # context: [batch,n_head,seq_len,d_v] attn: [batch,n_head,len_q,d_k]
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V)

        # context reshap to [batch,seq_len,head*d_v]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_head * self.d_v)

        context = self.fc(context)
        return nn.LayerNorm(self.d_model)(residual + context), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff) -> None:
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        '''
        @Desc    :   Forword Layer
        @Time    :   2020/11/06 14:44:07
        @Author  :   sam.qi 
        '''
        residual = inputs
        output = self.fc(inputs)
        # [batch,seq_len,d_model]
        return nn.LayerNorm(self.d_model)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_head) -> None:
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiheadAttention(
            d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_head)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, inputs, enc_self_mask):
        '''
        @Desc    :   inputs: [batch,seq_len,d_model]
                     enc_self_mask: [batch,seq_len,seq_len]
        @Time    :   2020/11/06 14:56:38
        @Author  :   sam.qi 
        '''

        context, attn = self.enc_self_attn(
            inputs, inputs, inputs, enc_self_mask)

        # output :[batch,seq_len,d_model]
        output = self.ff(context)
        return output, attn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, d_k, d_v, d_ff, max_seq_len=512, n_layer=6, n_head=8) -> None:
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, d_ff, n_head)
                                     for _ in range(n_layer)])

    def froward(self, inputs):
        '''
        @Desc    :   input: [batch,src_len]
        @Time    :   2020/11/06 15:06:13
        @Author  :   sam.qi 
        '''

        enc_outputs = self.src_emb(inputs)
        pos_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(inputs, inputs)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, d_ff) -> None:
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiheadAttention(
            d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_head)
        self.dec_enc_attn = MultiheadAttention(
            d_model=d_model, d_k=d_v, n_heads=n_head)
        self.fnn = FeedForward(d_model, d_ff=d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(
            dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(
            0, 1).cuda()  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(
            dec_inputs, dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(
            dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda()

        dec_enc_attn_mask = get_attn_pad_mask(
            dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, d_model, tgt_vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs)
        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

##########################################################################################################################################


def get_attn_pad_mask(seq_q, seq_k):
    '''
    @Desc    :   获取Mask矩阵
    @Time    :   2020/11/06 15:20:36
    @Author  :   sam.qi 
    '''

    batch, len_q = seq_q.size()
    batch, len_k = seq_k.size()
    # [batch,1,len_k]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch, len_q, len_k)


def get_attn_subsequence_mask(seq):
    '''
    @Desc    :   在Decoder中，屏蔽未来词
    @Time    :   2020/11/06 15:21:05
    @Author  :   sam.qi 
    '''

    #seq : [batch,seq_len]
    attn_shape = [seq.size(0), seq.size(1), seq(1)]
    subsequence_mask = np.triu(attn_shape, k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    # [batch_size, seq_len, seq_len]
    return subsequence_mask
