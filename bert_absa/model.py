#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/05 15:25:22
@Author  :   sam.qi
@Version :   1.0
@Desc    :   None
'''

import torch.nn as nn

from transformers import BertModel, AdamW, BertTokenizer


class EPClassify(nn.Module):
    def __init__(self,):
        super(EPClassify, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.linear = nn.Linear(768, 2)
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        bert_output = self.bert(x)
        linear_output = self.linear(bert_output)
        output = self.prob(linear_output)
        return output


class EPTrainer(object):
    def __init__(self) -> None:
        super().__init__()
        self.model = EPClassify()
        self.loss_fun = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

    def start(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        text_batch = ["I love Pixar.", "I don't care for Pixar."]
        encoding = tokenizer(text_batch, return_tensors='pt',
                             padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
