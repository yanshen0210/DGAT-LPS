import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, BatchNorm # noqa
from torch_geometric.utils import dropout_adj


class DGAT(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(DGAT, self).__init__()

        self.GConv1 = GATv2Conv(feature, 1024)
        self.bn1 = BatchNorm(1024)

        self.GConv2 = GATv2Conv(1024, 1024)
        self.bn2 = BatchNorm(1024)

        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(512, out_channel))


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        edge_index = dropout_adj(edge_index)
        x = self.GConv2(x, edge_index[0])
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x