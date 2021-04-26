#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/1 1:57 下午
# @Author  : Sam
# @Desc    :

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    '''
        当输入比较长，单独靠 Content Vector 无法获取整个意思时候，使用Attention Mechanism 来体通过 Decoder 更多信息
        主要是根据现在的Decoder Hidden State 去和Encoder outputs 中，那些与其有比较高的相关，根据相关性数值决定给Decoder 更多信息
        常见的 Attention 是用Decoder Hidden State 和Encoder outputs 计算 Dot Product，再对算出来的值做Softmax，最后根据Softmax值
        对Encoder outpus 做weight sum
    '''

    def __init__(self):
        super().__init__()

    def forward(self, encoder_output, decoder_hidden, source_mask=None):
        '''
            encoder_output: [batch,seq,hidden_dim * direction]
            decoder_hidden:[layers * direction,batch,hidden_dim]
            Decoder的每一层都和Output做Attention操作

        '''

        # encoder 变换，将Encoder output变成 [batch,seq,hidden]
        layer, batch, hidden = decoder_hidden.size()

        # Query * key
        decoder_hidden = decoder_hidden.permute(1, 0, 2)
        encoder_output_key = encoder_output.permute(0, 2, 1)
        attn = torch.matmul(decoder_hidden, encoder_output_key)

        # mask
        if source_mask is not None:
            source_mask = source_mask.view(batch, 1, -1)
            attn = attn.masked_fill(source_mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        # mask operation

        # compute value
        attn = torch.matmul(attn, encoder_output)

        # attn 转化
        attn = attn.view(layer, batch, hidden)

        return attn
