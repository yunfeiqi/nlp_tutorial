#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/1 1:32 下午
# @Author  : Sam
# @Desc    :

import torch.nn as nn
import torch
import random


class Seq2Seq(nn.Module):
    '''
            由Encoder 和Decoder组成
            接受输入传给Encoder
            将Encoder 的输出传给Decoder
            不断的将Decoder 输出回传给Decoder，进行解码
            解码完成，将 Decoder 输出回传
    '''

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, target, input_mask, target_mask, teacher_forcing_ratio):
        # input: [batch,sequence]
        # target: [batch,target len]
        # teacher_foring_ratio: 有多少几率使用Target
        batch_size = target.size(0)
        target_len = target.size(1)
        maxlen = torch.max(target_mask.sum(dim=-1)).item()
        vocab_size = self.decoder.cn_vocab_size

        # 存储结果
        outputs = torch.zeros(batch_size, target_len,
                              vocab_size).to(self.device)

        # ** 进行Encoder操作**
        encoder_output, hidden = self.encoder(input)
        # encoder_output 主要用于 Attension
        # encoder_hidden 用来初始化 Decoder的 Hidden
        # 因为Encoder是双向的，其维度是：[n_layer * direction,Batsh,Hid_dim] 所以需要进行转化
        # 转化方法就是将 direction 拼接在一起
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

        # 取 BOS
        input = target[:, 0]
        preds = []

        # ** 开始 Decoder **
        for step in range(1, maxlen):
            # decoder_output: [batch,CN_vocab_size]
            decoder_output, hidden = self.decoder(
                input, hidden, encoder_output, input_mask)
            outputs[:, step] = decoder_output
            teacher_force = random.random() <= teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            input = target[:,
                           step] if teacher_force and step < target_len else top1
            input = target[:, step]
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, input, target, source_mask):
        # TODO Use Beam Search

        batch_size = input.size(0)
        input_len = input.size(1)
        vocab_size = self.decoder.cn_vocab_size
        outputs = torch.zeros(batch_size, input_len,
                              vocab_size).to(self.device)

        # 开始 Encoder
        encoder_output, hidden = self.encoder(input)
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

        # 开始 Decoder
        # input: [batch,1]
        input = target[:, 0]
        preds = []
        for step in range(1, input_len):
            # decoder_output:[batch,1,hid_dim]
            decoder_output, hidden = self.decoder(
                input, hidden, encoder_output, source_mask)
            outputs[:, step] = decoder_output
            # top1:[batch,vocab]

            top1 = decoder_output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds