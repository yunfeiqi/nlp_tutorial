#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/12 18:07:16
@Author  :   sam.qi 
@Version :   1.0
@Desc    :   Naive RNN Demo 
'''

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self) -> None:
        super()
