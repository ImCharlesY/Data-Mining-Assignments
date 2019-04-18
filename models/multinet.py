#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiNet(torch.nn.Module):

  def __init__(self, nfeat, nhid, nlabel, dropout, nfeat1, module):
    super(MultiNet, self).__init__()

    self.net0 = module(nfeat, nhid, nlabel, dropout)
    self.net1 = module(nfeat1, nhid, nlabel, dropout)
    self.fc = nn.Linear(nlabel*2, nlabel, bias=False)

  def forward(self, x0, adj0, x1, adj1):
    h0 = self.net0(x0, adj0)
    h1 = self.net1(x1, adj1)
    x = torch.sigmoid(self.fc(torch.cat([h0, h1], 1)))
    return x, h0, h1
