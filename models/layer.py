#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GCNLayer(Module):

  def __init__(self, in_features, out_features, bias=True):
    super(GCNLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
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


class NGCNLayer(Module):

  def __init__(self, in_features, out_features, iterations, bias=True):
    super(NGCNLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.iterations = iterations
    self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
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
    for iteration in range(self.iterations):
      support = torch.spmm(adj, support)
    if self.bias is not None:
      return support + self.bias
    else:
      return support

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
           + str(self.in_features) + ' -> ' \
           + str(self.out_features) + ')' \
           + '(' + str(self.iterations) + ')'


class GraphAttentionLayer(Module):

  def __init__(self, in_features, out_features, dropout, alpha, concat=True, residual=False):
    super(GraphAttentionLayer, self).__init__()
    self.dropout = dropout
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = alpha
    self.concat = concat
    self.residual = residual

    self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
    if self.residual:
        self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
    self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
    self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, input, adj):
    # Too harsh to use the same dropout. TODO add another dropout
    # input = F.dropout(input, self.dropout, training=self.training)

    seq = torch.transpose(input, 0, 1).unsqueeze(0)
    seq_fts = self.seq_transformation(seq)

    f_1 = self.f_1(seq_fts)
    f_2 = self.f_2(seq_fts)
    logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
    coefs = F.softmax(self.leakyrelu(logits) * adj.to_dense(), dim=1)

    seq_fts = F.dropout(torch.transpose(seq_fts.squeeze(0), 0, 1), self.dropout, training=self.training)
    coefs = F.dropout(coefs, self.dropout, training=self.training)

    ret = torch.mm(coefs, seq_fts)

    if self.residual:
      if seq.size()[-1] != ret.size()[-1]:
        ret += torch.transpose(self.proj_residual(seq).squeeze(0), 0, 1)
      else:
        ret += input

    if self.concat:
      return F.elu(ret)
    else:
      return ret

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
           + str(self.in_features) + ' -> ' \
           + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
  """Special function for only sparse region backpropataion layer."""
  @staticmethod
  def forward(ctx, indices, values, shape, b):
    assert indices.requires_grad == False
    a = torch.sparse_coo_tensor(indices, values, shape)
    ctx.save_for_backward(a, b)
    ctx.N = shape[0]
    return torch.matmul(a, b)

  @staticmethod
  def backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_values = grad_b = None
    if ctx.needs_input_grad[1]:
      grad_a_dense = grad_output.matmul(b.t())
      edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
      grad_values = grad_a_dense.view(-1)[edge_idx]
    if ctx.needs_input_grad[3]:
      grad_b = a.t().matmul(grad_output)
    return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
  def forward(self, indices, values, shape, b):
    return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(Module):

  def __init__(self, in_features, out_features, dropout, alpha, concat=True):
    super(SpGraphAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = alpha
    self.concat = concat

    self.W = Parameter(torch.FloatTensor(in_features, out_features))
    self.a = Parameter(torch.FloatTensor(1, 2*out_features))

    self.dropout = nn.Dropout(dropout)
    self.leakyrelu = nn.LeakyReLU(self.alpha)
    self.special_spmm = SpecialSpmm()
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.W.data, gain=1.414)
    nn.init.xavier_uniform_(self.a.data, gain=1.414)

  def forward(self, input, adj):
    # Apply features transformation
    h = torch.mm(input, self.W)
    # h: N x out
    assert not torch.isnan(h).any()
    N = input.size()[0]

    # edge: 2*D x E
    edge = adj._indices()

    # Self-attention on the nodes - Shared attention mechanism
    edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
    # edge_e: E
    edge_e = self.leakyrelu(self.a.mm(edge_h).squeeze())

    # Do sparse softmax:
    # - Uses trick like logsumexp to confirm the stability (avoid underflow and overflow)
    # - softmax([1,3]) = [0.1192, 0.8808]
    # - softmax([-2,0]) = [0.1192, 0.8808]

    # Find the max of each row, edge_r_rm should shaped as (E,)
    # edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N]))
    # edge_e_rm = torch.sparse.max(edge_e, dim=1)
    # edge_e_rm = torch.max(edge_e) # For simple, just use the max of all
    # Subtract the max value of each row
    # edge_e = edge_e - edge_e_rm
    edge_e = -edge_e
    # Do exp
    edge_e = torch.exp(edge_e)
    assert not torch.isnan(edge_e).any()
    # Do sum
    # e_rowsum: N x 1
    e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), input.new_full([N, 1], fill_value=1))

    # edge_e: E
    edge_e = self.dropout(edge_e)

    # h_prime: N x out
    h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
    assert not torch.isnan(h_prime).any()
    
    # h_prime: N x out
    h_prime = h_prime.div(e_rowsum + 1e-9)
    assert not torch.isnan(h_prime).any()

    if self.concat:
      return F.elu(h_prime)
    else:
      return h_prime

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
           + str(self.in_features) + ' -> ' \
           + str(self.out_features) + ')'


class simpleAttnLayer(nn.Module):

  def __init__(self, nfeat, nmpattn):
    super(simpleAttnLayer, self).__init__()
    # Semantic-specific embedding transformation
    self.semantic_trans = nn.Linear(nfeat, nmpattn)
    # Semantic-level attention vector
    self.semantic_attn = nn.Linear(nmpattn, 1, bias=False)

  def forward(self, input):
    # Input should shaped as (N, nsemantic, nfeat)
    assert len(input.shape) == 3
    # Semantic-specific embedding transformation
    attn_multi_emb = torch.tanh(self.semantic_trans(input)) # now shaped as (N, nsemantic, nmpattn)
    # Apply semantic-level attention vector, get weight of each meta-path
    coeff_multi_emb = F.softmax(self.semantic_attn(attn_multi_emb), dim=1) # shaped as (N, nsemantic, 1)
    # Fuse semantic-specific embeddings
    final_emb = torch.sum(input * coeff_multi_emb, dim=1, keepdim=False) # shaped as (N, nfeat)
    return final_emb
