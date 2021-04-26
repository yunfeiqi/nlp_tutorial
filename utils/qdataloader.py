#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/14 12:41:47
@Author  :   sam.qi
@Version :   1.0
@Desc    :   自定义DataLoader，将给定的数据集转化成DataLoader
'''
from os import makedirs
import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset

from common.file import check_exist, read_all_lins


class BaseDataset(data.dataset.Dataset):
    def __init__(self, X, y=None) -> None:
        super().__init__()
        if not isinstance(X, torch.LongTensor):
            X = torch.LongTensor(X)

        if y is not None and not isinstance(y, torch.LongTensor):
            y = torch.LongTensor(y)
        self.data = X
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]


class DataAccess(object):
    '''
        数据接入
    '''

    def __init__(self) -> None:
        super().__init__()
        self.dataset = None

    def get_dataloader(self, batch_size, shuffle=True):
        if self.dataset is None:
            raise RuntimeError("dataset 未能构建构成功")

        return data.dataloader.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle)


class FileDataAccess(DataAccess):
    '''
        文件加载类
    '''

    def __init__(self, x_path, y_path=None, max_sentence=60) -> None:
        super().__init__()
        self.x_path = x_path
        self.y_path = y_path
        self.max_sentence = max_sentence
        self.dataset = None
        self.data_load()

    def data_load(self):
        """
        load data from file
        """

        if self.x_path is None or not check_exist(self.x_path):
            raise RuntimeError("指定文件不存在")

        lines = read_all_lins(self.x_path)

        if self.y_path is not None and check_exist(self.y_path):
            labels = read_all_lins(self.y_path)
        else:
            labels = None

        # split

        word2idx = {}
        idx2word = []
        lines = [line.split(" ") for line in lines]
        ids = []
        for line in lines:
            id_line = []
            for word in line:
                id = word2idx.get(word, None)
                if id is None:
                    id = len(idx2word)

                word2idx[word] = id
                idx2word.append(word)
                id_line.append(id)

            if len(id_line) > self.max_sentence:
                id_line = id_line[:self.max_sentence]
            else:
                pad_len = self.max_sentence - len(id_line)
                for i in range(pad_len):
                    id_line.append(0)

            ids.append(id_line)

        self.data = ids
        self.labels = labels
        self.dataset = BaseDataset(self.data, self.labels)


class MemoryDataAccess(DataAccess):
    '''
        文件加载类
    '''

    def __init__(self, x, y=None) -> None:
        super().__init__()
        self.dataset = BaseDataset(x, y)


if __name__ == "__main__":
    fds = FileDataAccess("C:\\Users\\59442\Documents\\training_nolabel.txt")
    loader = fds.get_dataloader(batch_size=2, shuffle=False)
    for i, d in enumerate(loader):
        print(d)
        break
