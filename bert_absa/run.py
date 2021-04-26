#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/04 09:38:26
@Author  :   sam.qi
@Version :   1.0
@Desc    :   None
'''

from transformers import BertTokenizer, BertForPreTraining, BertConfig, BertModel


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForPreTraining.from_pretrained("bert-base-chinese")

inputs = tokenizer(
    "4日下午4时许，习近平来到京西宾馆，在热烈的掌声中亲切接见全体会议代表，并同大家合影留念", return_tensors="pt")

outputs = model(**inputs)

prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits
