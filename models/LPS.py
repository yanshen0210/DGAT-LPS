import logging
import time
import warnings
import torch
from torch import nn
from torch import optim
import models
import random
import numpy as np


def LPS(args, datasets_train):
    samples = np.array(range(args.train_sample * args.fault_num))
    train, train_mask, label_mask = [], [], []
    edge_train = datasets_train.edge_index

    for i in range(args.fault_num):
        sample_c = samples[i * args.train_sample:(i + 1) * args.train_sample]
        random.seed(100)
        random.shuffle(sample_c)
        train += torch.LongTensor(sample_c[:args.train_num])  # samples with true labels
        train_mask += torch.LongTensor(sample_c[:args.train_num])  # samples with true and fake labels
        label_mask += torch.LongTensor(np.zeros(args.train_num) + i)  # true and fake labels
        for j in sample_c[:args.train_num]:
            for k in range(args.k_value):
                train_mask += edge_train[1][args.k_value * j + k].unsqueeze(0)
                label_mask += torch.LongTensor(np.zeros(1) + i)
    train_mask = torch.LongTensor(train_mask)
    label_mask = torch.LongTensor(label_mask)
    idx_train = torch.LongTensor(train)

    # calculate acc of fake labels
    labels_true = datasets_train.y[train_mask]
    true = torch.eq(label_mask, labels_true).float().sum().item()
    acc = (true - args.train_num * args.fault_num) / (args.k_value * args.train_num * args.fault_num)
    logging.info('\n Accuracy of the LPS: {:.4f} \n'.format(acc))

    return idx_train, train_mask, label_mask
