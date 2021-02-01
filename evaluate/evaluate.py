#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/27 3:29 下午
# @Author  : Sam
# @Desc    : 基于Pytorch的评价指标



def evaluate(pred, target):
    '''
        pred: [batch]
        target:[target]
    '''
    # Accuracy
    size = pred.size()[0]
    right = pred == target
    right = right.sum()
    acc = right/size

    # F1
    # TP predict 和 label 同时为1
    TP = ((pred == 1) & (target == 1)).cpu().sum()
    # TN predict 和 label 同时为0
    TN = ((pred == 0) & (target == 0)).cpu().sum()
    # FN predict 0 label 1
    FN = ((pred == 0) & (target == 1)).cpu().sum()
    # FP predict 1 label 0
    FP = ((pred == 1) & (target == 0)).cpu().sum()

    p = TP / ((TP + FP)+0.00001)
    r = TP / ((TP + FN)+0.00001)
    F1 = 2 * r * p / (r + p+0.00001)
    return acc, F1
