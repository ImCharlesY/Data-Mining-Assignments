#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GCNLayer(Module):

  def __init__(self, in_channels, out_channels, bias=True):
    super(GCNLayer, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.weight = Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
    if bias:
      self.bias = Parameter(torch.FloatTensor(out_channels))
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
           + str(self.in_channels) + ' -> ' \
           + str(self.out_channels) + ')'


class NGCNLayer(Module):

  def __init__(self, in_channels, out_channels, iterations, bias=True):
    super(NGCNLayer, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.iterations = iterations
    self.weight = Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
    if bias:
      self.bias = Parameter(torch.FloatTensor(out_channels))
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
    for iteration in range(self.iterations):
      support = torch.spmm(adj, support)
    if self.bias is not None:
      return support + self.bias
    else:
      return support

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
           + str(self.in_channels) + ' -> ' \
           + str(self.out_channels) + ')' \
           + '(' + str(self.iterations) + ')'
