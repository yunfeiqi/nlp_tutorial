#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/1 1:58 下午
# @Author  : Sam
# @Desc    :

import re
import json
import torch
import os
import numpy as np
from torch.utils.data import Dataset


class LabelTransform(object):
    def __init__(self, size, pad) -> None:
        self.size = size
        self.pad = pad

    def __call__(self, label):
        mask = np.ones(label.shape[0])

        label = np.pad(
            label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        mask = np.pad(
            mask, (0, (self.size - mask.shape[0])), mode='constant', constant_values=0)

        return label, mask


class EN2CnDataset(Dataset):
    '''
        定义英文转中文Dataset
    '''

    def __init__(self, root, max_output_len, set_name) -> None:
        super().__init__()
        self.root = root

        # 加载分词字典
        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')

        # 加载数据集
        self.data = []
        with open(os.path.join(self.root, f'{set_name}.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(line)

        print(f'{set_name} dataset size: {len(self.data)}')

        # 统计
        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)

        # 将所有输入和输出对齐
        self.transform = LabelTransform(
            max_output_len, self.word2int_en['<PAD>'])

    def get_dictionary(self, lang):
        '''
            根据不同的语言，加载不同的Vocab Map
        '''
        with open(os.path.join(self.root, f'word2int_{lang}.json'), 'r', encoding='utf-8') as f:
            word2int = json.load(f)

        with open(os.path.join(self.root, f"int2word_{lang}.json"), 'r', encoding='utf-8') as f:
            int2word = json.load(f)

        return word2int, int2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentences = self.data[index]
        # 拆分中英文
        sentences = re.split('[\t]', sentences)
        sentences = list(filter(None, sentences))
        assert len(sentences) == 2

        # 特殊子元
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']

        # 在开头添加 BOS 结尾添加EOS，OOV 使用 UNK
        en, cn = [BOS], [BOS]
        sentence = re.split(" ", sentences[0])
        sentence = list(filter(None, sentence))
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        sentence = re.split(" ", sentences[1])
        sentence = list(filter(None, sentence))
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)

        en, cn = np.array(en), np.array(cn)
        # 将句子补齐
        en, en_mask = self.transform(en)
        cn, cn_mask = self.transform(cn)
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)
        en_mask, cn_mask = torch.LongTensor(en_mask), torch.LongTensor(cn_mask)

        return en, cn, en_mask, cn_mask
