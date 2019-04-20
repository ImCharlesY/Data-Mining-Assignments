#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import simpleAttnLayer


class MultiNet(torch.nn.Module):

  def __init__(self, nfeat, nhid, nmid, nlabel, dropout, nfeat1, module, nmpattn=128):
    super(MultiNet, self).__init__()
    self.dropout = dropout

    self.net0 = module(nfeat, nhid, nmid, dropout)
    self.net1 = module(nfeat1, nhid, nmid, dropout)

    self.semantic_attn = simpleAttnLayer(nmid, nmpattn)
    self.classifier = nn.Linear(nmid, nlabel, bias=False)

  def forward(self, x, adj):
    h0 = self.net0(x[0], adj[0])
    h1 = self.net1(x[1], adj[1])
    h = torch.cat([h0.unsqueeze(1), h1.unsqueeze(1)], 1)
    final_emb = self.semantic_attn(h)
    final_emb = F.dropout(final_emb, self.dropout, training=self.training)
    out = torch.sigmoid(self.classifier(final_emb))    
    return out
