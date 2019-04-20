#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import GCNLayer


class GCN(nn.Module):
  def __init__(self, nfeat, nhid, nlabel, dropout):
    super(GCN, self).__init__()

    self.dropout = dropout

    self.gc1 = GCNLayer(nfeat, nhid)
    self.gc2 = GCNLayer(nhid, nlabel)

  def forward(self, x, adj):
    # x = F.dropout(x, self.dropout, training=self.training)
    x = F.relu(self.gc1(x, adj))
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.sigmoid(self.gc2(x, adj))
    return x
