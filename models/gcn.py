import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nlabel, dropout, crf=False):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.crf = crf

        # self.gc1 = GraphConvolution(nfeat, 4096)
        # self.mid = nn.ModuleList([
        #     GraphConvolution(4096, 2048),
        #     GraphConvolution(2048, 1024), 
        # ])
        # self.gc2 = GraphConvolution(1024, nlabel)
        self.gc1 = GraphConvolution(nfeat, nhid)
        if self.crf:
            self.gc2 = nn.ModuleList([GraphConvolution(nhid, 2) for _ in range(nlabel)])
        else:
            self.gc2 = GraphConvolution(nhid, nlabel)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # for m in self.mid:
        #     x = F.relu(m(x, adj))
        #     x = F.dropout(x, self.dropout, training=self.training)
        if self.crf:
            x = torch.cat([gc(x, adj).unsqueeze(1) for gc in self.gc2], 1)
        else:
            x = torch.sigmoid(self.gc2(x, adj))
        return x
