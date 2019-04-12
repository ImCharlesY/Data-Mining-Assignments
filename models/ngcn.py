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

  def __init__(self, nfeat, nlabel, dropout, layers=[16,16,16]):
    super(NGCN, self).__init__()

    self.dropout = dropout

    self.layers = layers
    self.order = len(self.layers)

    self.main_layers = nn.ModuleList([NGCNLayer(nfeat, self.layers[i-1], i) for i in range(1, self.order+1)])
    self.fc = nn.Linear(sum(self.layers), nlabel)

  def forward(self, x, adj):
    x = F.relu(torch.cat([layer(x, adj) for layer in self.main_layers],dim=1))
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.sigmoid(self.fc(x))        
    return x
