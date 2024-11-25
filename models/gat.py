import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GraphConv, GINConv, GATConv
import torch.nn.functional as F
import copy
from torch_scatter import scatter_mean, scatter_max, scatter_add


class GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, pooling='scatter_mean', dropout=0.5):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_feats, hid_feats, heads=8, dropout=dropout)
        self.conv2 = GATConv(hid_feats * 8, out_feats, heads=8, concat=False, dropout=dropout)
        self.pooling = pooling

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if self.pooling == 'scatter_mean':
            x = scatter_mean(x, data.batch, dim=0)
        elif self.pooling == 'scatter_max':
            x = scatter_max(x, data.batch, dim=0)
        elif self.pooling == 'scatter_add':
            x = scatter_add(x, data.batch, dim=0)
        elif self.pooling == 'global_mean':
            x = global_mean_pool(x, data.batch)
        elif self.pooling == 'global_max':
            x = global_max_pool(x, data.batch)
        elif self.pooling == 'mean_max':
            x_mean = global_mean_pool(x, data.batch)
            x_max = global_max_pool(x, data.batch)
            x = torch.cat((x_mean, x_max), 1)
        elif self.pooling == 'scatter_mean_max':
            x_mean = scatter_mean(x, data.batch, dim=0)
            x_max = scatter_add(x, data.batch, dim=0)
            x = torch.cat([x_mean, x_max], 1)
        elif self.pooling == 'root':
            rootindex = data.rootindex
            root_extend = torch.zeros(len(data.batch), 768).to(x.device)
            batch_size = max(data.batch) + 1
            for num_batch in range(batch_size):
                index = (torch.eq(data.batch, num_batch))
                root_extend[index] = x[rootindex[num_batch]]
            x = root_extend
        else:
            assert False, "Something wrong with the parameter --pooling"
        return x
