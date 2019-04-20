#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import torch


def accuracy(output, labels, threshold=0.5):
  preds = torch.ge(output.float(), threshold).float()
  correct = preds.eq(labels).float()
  correct = correct.sum()
  return correct / labels.numel()
    

def f1_score(output, labels, eps=1e-9, threshold=0.5):
  output = torch.ge(output.float(), threshold).float()
  labels = labels.float()
  true_positive = (output * labels).sum(dim=1)
  precision = true_positive.div(output.sum(dim=1).add(eps))
  recall = true_positive.div(labels.sum(dim=1).add(eps))
  return torch.mean((2 * precision * recall).div(precision + recall + eps))


def f1_loss(output, labels, eps=1e-9):
  output = output.float()
  labels = labels.float()
  true_positive = (output * labels).sum(dim=1)
  precision = true_positive.div(output.sum(dim=1).add(eps))
  recall = true_positive.div(labels.sum(dim=1).add(eps))
  f1 = torch.mean((2 * precision * recall).div(precision + recall + eps))
  return 1 - f1


def hamming_loss(output, labels):
  output = output.float()
  labels = labels.float()
  hamming = torch.mean((1 - output) * labels + (1 - labels) * output)
  return hamming  
