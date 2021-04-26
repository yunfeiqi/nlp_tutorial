#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/27 3:32 下午
# @Author  : Sam
# @Desc    :

PLACE_HOLDER = "*********************************************************"
RED = "\033[0;32;31m"
GREEN = "\033[0;32;32m"
YELLOW = "\033[1;33m"
NONE = "\033[m"


def add_placeholder(msg):
    return "\n{}\n{}\n{}\n".format(PLACE_HOLDER, msg, PLACE_HOLDER)


def add_red(msg):
    return "\n{}{}{}\n".format(RED, msg, NONE)


def add_green(msg):
    return "\n{}{}{}\n".format(GREEN, msg, NONE)


def add_yellow(msg):
    return "\n{}{}{}\n".format(YELLOW, msg, NONE)


def exec_print(msg, logger=None):
    '''
        打印消息
    :param msg:
    :param logger:
    :return:
    '''
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def info(msg, logger=None):
    msg = add_placeholder(msg)
    msg = add_green(msg)
    exec_print(msg, logger)


def debug(msg, logger=None):
    msg = add_placeholder(msg)
    msg = add_yellow(msg)
    exec_print(msg, logger)


def score(scores, logger=None):
    '''
        打印指标
    :param scores: 字典类型
    :return: None
    '''
    msg = ""

    if isinstance(scores, str):
        msg = scores

    elif isinstance(scores, dict):
        for key, value in scores.items():
            msg += "| {}:{}".format(key, value)
    elif isinstance(scores, list):
        msg = ",".join(scores)
    else:
        msg = ""
    exec_print(msg, logger)
