import os
import pandas as pd
from datasets.Generator import Gen_graph
from tqdm import tqdm
import numpy as np
import random
from scipy.io import loadmat


# generate Training Dataset and Testing Dataset
def get_files(args):
    sub_dir = []
    data_train = []
    lab_train = []
    data_test = []
    lab_test = []

    root = os.path.join(args.data_dir)  # the location of the dataset
    file_name = os.listdir(root)  # all fault modes
    for j in file_name:
        sub_dir.append(os.path.join(root, j))

    for i in tqdm(range(len(sub_dir))):
        data1_train, lab1_train, data1_test, lab1_test = data_load(args, sub_dir[i], label=i)
        data_train += data1_train
        lab_train += list(lab1_train)
        data_test += data1_test
        lab_test += list(lab1_test)

    graphset_train = Gen_graph(args, data_train, lab_train)
    graphset_test = Gen_graph(args, data_test, lab_test)
    return graphset_train, graphset_test


def data_load(args, root, label):
    fl = loadmat(root)['Channel_1']
    fl = fl.reshape(-1,)

    data = []
    start, end = 0, args.sample_length

    for i in range(args.sample_size):
        x = fl[start:end]
        x = (x - x.min()) / (x.max() - x.min())  # normalization
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]

        data.append(x)
        start += args.sample_length
        end += args.sample_length
    random.seed(100)
    random.shuffle(data)
    data_train = data[:args.train_sample]
    data_test = data[args.train_sample:args.sample_size]
    label_train = np.zeros(args.train_sample) + label
    label_test = np.zeros(args.sample_size-args.train_sample) + label

    return data_train, label_train, data_test, label_test