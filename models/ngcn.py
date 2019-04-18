#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import NGCNLayer


class NGCN(torch.nn.Module):

  def __init__(self, nfeat, nhid, nlabel, dropout):
    super(NGCN, self).__init__()

    self.dropout = dropout
    self.order = 3

    self.main_layers = nn.ModuleList([NGCNLayer(nfeat, nhid, i) for i in range(1, self.order+1)])
    self.fc = nn.Linear(nhid*self.order, nlabel)

  def forward(self, x, adj):
    x = F.relu(torch.cat([layer(x, adj) for layer in self.main_layers],dim=1))
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.sigmoid(self.fc(x))        
    return x
