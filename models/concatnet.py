#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gcn import GCN
from models.ngcn import NGCN
from models.gccn import GCCN


class ConcatNet(torch.nn.Module):

  def __init__(self, nfeat, nhid, nlayer, dropout, postnet):
    super(ConcatNet, self).__init__()

    self.dropout = dropout

    self.fc1 = nn.Linear(nfeat, nhid)
    self.mid = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(nlayer)])
    self.postnet = postnet

  def forward(self, x, adj):
    x, em = x
    x = F.relu(self.fc1(x))
    x = F.dropout(x, self.dropout, training=self.training)
    for mid in self.mid:
        x = F.relu(mid(x))
        x = F.dropout(x, self.dropout, training=self.training)
    x = torch.cat([x, em], 1)
    x = self.postnet(x, adj)
    return x
