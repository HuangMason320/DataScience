import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from torch import Tensor
# from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from dgl.nn.pytorch import GraphConv

# 第三份code中的模型定義部分
class Net(torch.nn.Module):
    def __init__(self, num_features, hidden, dropout, num_classes):
        super(Net, self).__init__()
        self.crd = CRD(num_features, hidden, dropout)
        self.cls = CLS(hidden, num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    # def forward(self, data):
    #     x, edge_index = data.x, data.edge_index
    #     x = self.crd(x, edge_index, data.train_mask)
    #     x = self.cls(x, edge_index, data.train_mask)
    #     return x
    def forward(self, data, features):
        data.ndata['x'] = features
        x = data.ndata['x']
        edge_index = data.edges()
        train_mask = data.ndata['train_mask']
        x = self.crd(x, edge_index, train_mask)
        x = self.cls(x, edge_index, train_mask)
        return x


class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True) 
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
    
# class Net(torch.nn.Module):
#     def __init__(self, features, num_classes, args):
#         super(Net, self).__init__()
#         self.crd = CRD(features.shape[1], args.hidden, args.dropout)  # 使用 features.shape[1] 而不是 features
#         self.cls = CLS(args.hidden, num_classes)

#     def reset_parameters(self):
#         self.crd.reset_parameters()
#         self.cls.reset_parameters()

    # def forward(self, data, features):
    #     data.ndata['x'] = features
    #     x = data.ndata['x']
    #     edge_index = data.edges()
    #     train_mask = data.ndata['train_mask']
    #     x = self.crd(x, edge_index, train_mask)
    #     x = self.cls(x, edge_index, train_mask)
    #     return x
    
    
    
    
# class YourGNNModel(nn.Module):
#     """
#     TODO: Use GCN model as reference, implement your own model here to achieve higher accuracy on testing data
#     """
#     def __init__(self, in_size, hid_size, out_size):
#         super().__init__()
    
#     def forward(self, g, features):
#         pass

