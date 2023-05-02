#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os

from utils.logger import setlogger
import logging
from utils.train_test_graph import train_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--sample_length', type=int, default=10000, help='points of a sample')
    parser.add_argument('--data_dir', type=str, default="./data", help='the directory of the data')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./results', help='the directory to save the model')

    # core parameters
    parser.add_argument('--fault_num', type=int, default=5, help='number of fault types')
    parser.add_argument('--sample_size', type=int, default=200, help='the number of samples for each fault type')
    parser.add_argument('--train_sample', type=int, default=50, help='the number of train samples for each fault type')
    parser.add_argument('--train_num', type=int, default=5, help='the number of labeled train samples ')
    parser.add_argument('--lps', type=bool, default=True, help='Whether add the LPS')
    parser.add_argument('--k_value', type=int, default=5, help='the k value of KNN')


    # optimization information
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='100,150', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=200, help='max number of epoch')
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    if args.lps:
        sub_dir = 'DGAT_LPS'
    else:
        sub_dir = 'DGAT'
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args)
    trainer.setup()
    trainer.train_test()


