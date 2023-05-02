import torch
from math import sqrt
import numpy as np
from torch_geometric.data import Data


def KNN_classify(k, X_set, x):
    """
    k:number of neighbours
    X_set: the datset of x
    x: to find the nearest neighbor of data x
    """
    distances = [sqrt(np.sum((x_compare-x)**2)) for x_compare in X_set]   # calculate the distance
    nearest = np.argsort(distances)  # sort the number by distance
    node_index = [i for i in nearest[1:k+1]]  # choose the nearest k number
    top_k = [X_set[i] for i in nearest[1:k+1]]  # choose the nearest k samples

    return node_index, top_k


def KNN_attr(args, data):
    edge_raw0 = []
    edge_raw1 = []

    for i in range(len(data)):
        x = data[i]

        node_index, top_k = KNN_classify(args.k_value, data, x)
        local_index = np.zeros(args.k_value)+i
        edge_raw0 = np.hstack((edge_raw0, local_index))  # center nodes
        edge_raw1 = np.hstack((edge_raw1, node_index))   # neighbor nodes

    edge_index = [edge_raw0, edge_raw1]
    return edge_index


def Gen_graph(args, data, label):
    node_edge = KNN_attr(args, data)  # return the edge of graph
    data = torch.tensor(data, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze(0)
    node_edge = torch.tensor(node_edge, dtype=torch.long)
    graph = Data(x=data, y=label, edge_index=node_edge)
    return graph
