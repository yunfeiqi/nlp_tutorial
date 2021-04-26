#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/14 12:51:04
@Author  :   sam.qi
@Version :   1.0
@Desc    :   文件相关操作
'''

import os


def check_exist(path):
    return os.path.exists(path)


def check_is_file(path):
    return os.path.isfile(path)


def read_all_lins(path, encoding='utf-8'):
    if not (check_exist(path) and check_is_file(path)):
        raise RuntimeError("输入路径{}不存在或者不是文件".format(path))

    with open(path, encoding=encoding) as f:
        lines = f.readlines()

    return lines
