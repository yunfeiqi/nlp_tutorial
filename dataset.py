#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/09 11:36:45
@Author  :   sam.qi 
@Version :   1.0
@Desc    :   数据源
'''

import torch.utils.data as Data


class TrainDataSet(Data.Dataset):
    def __init__(self, enc_input, dec_input, dec_output) -> None:
        super(TrainDataSet, self).__init__()
        self.enc_input = enc_input
        self.dec_input = dec_input
        self.dec_output = dec_output

    def __len__(self):
        return self.enc_input.shape()[0]

    def __getitem__(self, index):
        return self.enc_input[index], self.dec_input[index], self.dec_output[index]
