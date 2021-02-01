#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/1 2:00 下午
# @Author  : Sam
# @Desc    :

import torch
from collections import OrderedDict


def load_model(model, path):
    '''
        加载模型
    '''
    state_dict = torch.load(path, map_location=torch.device("cpu"))

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model


def schedule_sampling(step):
    '''
        Schedule Sampling 策略
    '''
    if step < 2500:
        return 1
    return 1 / (step - 2500)


def token2sentence(output, int2word):
    '''
        将数字转化为字
    '''
    sentences = []
    for tokens in output:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == "<EOS>":
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences

def save_model(model, store_model_path, step):
    torch.save(model.state_dict(), f'{store_model_path}/model_{step}.ckpt')



def infinite_iter(data_loader):
    '''
        无限迭代器
    '''

    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)

