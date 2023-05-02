#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import warnings
import torch
from torch import nn
from torch import optim
import models
import numpy as np

from datasets.data_load import get_files
from models.LPS import LPS
from models.DGAT import DGAT


class train_utils(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        self.datasets_train, self.datasets_test = get_files(args)

        # Define the model
        feature = int(args.sample_length/2)
        self.model = DGAT(feature=feature, out_channel=args.fault_num)

        # Define the optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                    weight_decay=args.weight_decay)

        # Define the learning rate decay
        steps = [int(step) for step in args.steps.split(',')]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_test(self):
        args = self.args

        idx_train, train_mask, label_mask = LPS(args, self.datasets_train)  # make the fake labels
        inputs_train = self.datasets_train.to(self.device)
        inputs_test = self.datasets_test.to(self.device)
        labels_train = inputs_train.y
        labels_test = inputs_test.y
        idx_train = idx_train.to(self.device)
        train_mask = train_mask.to(self.device)
        label_mask = label_mask.to(self.device)

        for epoch in range(args.max_epoch):

            # training process
            self.model.train()
            with torch.set_grad_enabled(True):
                logits = self.model(inputs_train)
                if args.lps:
                    loss = self.criterion(logits[train_mask], label_mask)
                    pred = logits[train_mask].argmax(dim=1)
                    correct = torch.eq(pred, labels_train[train_mask]).float().sum().item()
                    trian_loss = loss.item()
                    trian_acc = correct / (len(label_mask))
                else:
                    loss = self.criterion(logits[idx_train], labels_train[idx_train])
                    pred = logits[idx_train].argmax(dim=1)
                    correct = torch.eq(pred, labels_train[idx_train]).float().sum().item()
                    trian_loss = loss.item()
                    trian_acc = correct / (args.train_num * args.fault_num)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
                else:
                    logging.info('current lr: {}'.format(args.lr))

            # test process
            self.model.eval()
            logits = self.model(inputs_test)  # 输入测试集
            loss = self.criterion(logits, labels_test)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, labels_test).float().sum().item()
            test_loss = loss.item()
            test_acc = correct / ((args.sample_size - args.train_sample) * args.fault_num)

            logging.info('Epoch: {} train-Loss: {:.6f} train-Acc: {:.4f} \n test-Loss: {:.6f} test-Acc: {:.4f}'.format(
                epoch, trian_loss, trian_acc, test_loss, test_acc))


