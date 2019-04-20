#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import GraphAttentionLayer, SpGraphAttentionLayer, simpleAttnLayer


class GAT(nn.Module):

  def __init__(self, nfeat, nhid, nlabel, dropout, alpha=0.2, nheads=8, sparse=True):
    super(GAT, self).__init__()
    self.dropout = dropout
    self.layerModule = GraphAttentionLayer if not sparse else SpGraphAttentionLayer

    self.attentions = [self.layerModule(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)

    self.out_att = self.layerModule(nhid * nheads, nlabel, dropout=dropout, alpha=alpha, concat=False)

  def forward(self, x, adj):
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
    x = F.dropout(x, self.dropout, training=self.training)
    x = F.elu(self.out_att(x, adj))
    x = torch.sigmoid(x)
    return x


class HAT(nn.Module):

  def __init__(self, nfeat, nhid, nlabel, dropout, alpha=0.2, nheads=8, nsemantic=2, nmpattn=128, sparse=True):
    super(HAT, self).__init__()
    self.dropout = dropout
    self.layerModule = GraphAttentionLayer if not sparse else SpGraphAttentionLayer

    self.attentions = []
    for i in range(nsemantic):
      attentions = [self.layerModule(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
      for j, attention in enumerate(attentions):
        self.add_module('attention_{}_{}'.format(i, j), attention)
      self.attentions.append(attentions)
    
    self.semantic_attn = simpleAttnLayer(nhid * nheads, nmpattn)
    self.classifier = nn.Linear(nhid * nheads, nlabel, bias=False)

  def forward(self, x, adjs):
    if len(adjs) != len(self.attentions):
      raise ValueError('Number of semantic error.')
    # input data shaped as (N, F)
    x = F.dropout(x, self.dropout, training=self.training)
    meta_path_emb = []
    for adj, attentions in zip(adjs, self.attentions):
      # h shaped as (N, nhead * nhid)
      h = torch.cat([att(x, adj) for att in attentions], dim=1)
      meta_path_emb.append(h.unsqueeze(1))

    # multi_emb shaped as (N, nsemantic, nhead * nhid)
    multi_emb = torch.cat(meta_path_emb, dim=1)
    final_emb = self.semantic_attn(multi_emb)

    # Apply final classifier
    final_emb = F.dropout(final_emb, self.dropout, training=self.training)
    out = self.classifier(final_emb)
    out = torch.sigmoid(out)
    return out
