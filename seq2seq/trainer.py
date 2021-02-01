#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/1 1:32 下午
# @Author  : Sam
# @Desc    :


import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.helper import save_model
from utils.helper import infinite_iter
from utils.helper import schedule_sampling
from utils.helper import token2sentence
from dataset.dataset import EN2CnDataset

from model.encoder import RNNEncoder
from model.decoder import RNNDecoder
from model.Seq2Seq import Seq2Seq

device = "cuda:2" if torch.cuda.is_available() else "cpu"


def train(model, optimizer, train_iter, loss_function, total_steps, num_epochs):
    '''
        训练模型
    '''
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0

    for step in range(num_epochs):
        optimizer.zero_grad()

        sources, target, source_mask, target_mask = next(train_iter)
        sources, target, source_mask, target_mask = sources.to(device), target.to(
            device), source_mask.to(device), target_mask.to(device)

        outputs, preds = model(sources, target, source_mask,
                               target_mask, schedule_sampling(step))

        # 忽略 Target 的第一个Token，因为它是BOS
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        target = target[:, 1:].reshape(-1)
        loss = loss_function(outputs, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}".format(
                total_steps + step + 1, loss_sum, np.exp(loss_sum)), end=" ")
            losses.append(loss_sum)
            loss_sum = 0
    return model, optimizer, losses


def testing(model, dataloader, loss_function):
    model.eval()
    loss_sum = 0
    bleu_score = 0
    n = 0

    result = []
    for source, target, source_mask, target_mask in dataloader:
        source, target, source_mask, target_mask = source.to(
            device), target.to(device), source_mask.to(device), target_mask.to(device)
        batch_size = source.size(0)
        output, preds = model.inference(source, target, source_mask)
        output = output[:, 1:].reshape(-1, output.size(2))
        target = target[:, 1:].reshape(-1)

        loss = loss_function(output, target)
        loss_sum += loss.item()

        # 将预测结果转为文字
        targets = target.view(source.size(0), -1)
        preds = token2sentence(preds, dataloader.dataset.int2word_cn)
        sources = token2sentence(
            source.to("cpu").numpy(), dataloader.dataset.int2word_en)
        targets = token2sentence(
            targets.to("cpu").numpy(), dataloader.dataset.int2word_cn)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))

        n += batch_size
    return loss_sum / len(dataloader), result


def build_model(config, en_vocab_size, cn_vocab_size):
    # 构建模型实例
    encoder = RNNEncoder(en_vocab_size, config["emb_dim"],
                         config["hid_dim"], config["n_layers"], config["dropout"])
    decoder = RNNDecoder(cn_vocab_size, config["emb_dim"], config["hid_dim"],
                         config["n_layers"], config["dropout"], config["isatt"])
    model = Seq2Seq(encoder, decoder, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model = model.to(device)
    return model, optimizer


# ------------------------------- 训练&测试模型入口 --------------------------------

def train_process(config):
    # 准备训练数据
    train_dataset = EN2CnDataset(
        config["data_path"], config["max_output_len"], 'training')
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True)
    train_iter = infinite_iter(train_loader)
    # 准备验证数据
    val_datset = EN2CnDataset(
        config["data_path"], config["max_output_len"], 'validation')
    val_loader = DataLoader(val_datset, batch_size=1)

    # 构建模型实例
    model, optimizer = build_model(
        config, train_dataset.en_vocab_size, val_datset.cn_vocab_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while total_steps < config["num_steps"]:
        # 训练模型
        model, optimizer, loss = train(
            model, optimizer, train_iter, loss_function, total_steps, config["summary_steps"])
        train_losses += loss

        # 验证模型
        val_loss, result = testing(
            model, val_loader, loss_function)
        val_losses.append(val_loss)

        total_steps += config["summary_steps"]
        print("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}".format(
            total_steps, val_loss, np.exp(val_loss)))

        # 保存模型
        if total_steps % config["store_steps"] == 0 or total_steps >= config["num_steps"]:
            save_model(model, optimizer, config["store_model_path"], total_steps)
            with open(f'{config["store_model_path"]}/output_{total_steps}.txt', 'w') as f:
                for line in result:
                    print(line, file=f)

    return train_losses, val_losses


def test_process(config):
    test_dataset = EN2CnDataset(
        config.data_path, config.max_output_len, 'testing')
    test_loader = DataLoader(test_dataset, batch_size=1)
    model, optimizer = build_model(
        config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
    print("Finish build model")
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    test_loss, result = testing(model, test_loader, loss_function)
    # 保存结果
    with open("./test_output.txt", 'w') as f:
        for line in result:
            print(line, file=f)

    return test_loss


def train_entry(config):
    # ------------------------------- 开始训练模型 --------------------------------
    # print('config:\n', vars(config))
    train_losses, val_losses = train_process(config)


if __name__ == '__main__':
    # read config
    with open("config/conf_proj.json", 'r', encoding="utf-8") as f:
        conf = json.load(f)

    train_entry(conf)
