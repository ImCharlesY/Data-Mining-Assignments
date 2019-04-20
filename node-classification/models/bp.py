#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class BP(nn.Module):
  def __init__(self, nfeat, nhid, nlabel, dropout):
    super(BP, self).__init__()

    self.dropout = dropout

    self.fc1 = nn.Linear(nfeat, nhid)
    self.mid = nn.Linear(nhid, nhid)
    self.fc2 = nn.Linear(nhid, nlabel)

  def forward(self, x, _):
    x = F.relu(self.fc1(x))
    x = F.dropout(x, self.dropout, training=self.training)
    x = F.relu(self.mid(x))
    x = F.dropout(x, self.dropout, training=self.training)   
    x = torch.sigmoid(self.fc2(x))
    return x
