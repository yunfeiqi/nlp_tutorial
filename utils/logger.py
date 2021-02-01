#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/10/23 15:38:44
@Author  :   sam.qi
@Version :   1.0
@Desc    :   日志打印配置类
'''

from logging.handlers import RotatingFileHandler
import logging.handlers
import logging
import os
import sys
sys.path.append(os.getcwd())

# if not os.path.exists("logs"):
#     os.makedirs("logs")


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    # 配置日志格式
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(
        '%(levelname)s: %(asctime)s - %(filename)s:%(lineno)d - message: %(message)s')
    # 此段代码将日志打印到终端，否则可注释掉
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # # 设置日志的记录等级-以下设置将日志输出到文件中
    # 创建日志记录器，指明日志保存的路径、每个日志文件的最大大小、保存的日志文件个数上限
    # file_log_handler = RotatingFileHandler(
    #     log_file, maxBytes=1024 * 1024 * 100, backupCount=10)
    # 为刚创建的日志记录器设置日志记录格式
    # file_log_handler.setFormatter(formatter)
    # 为全局的日志工具对象（flask app使用的）添加日志记录器
    # logger.addHandler(file_log_handler)

    return logger


logger = get_logger()
