#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import os
import gc
import sys
import time
import argparse
import numpy as np
import pandas as pd

import torch

from utils.utils import load_data
from utils.measure import accuracy, f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', 
                    type=str,
                    default='data',
          help='Dataset directory. Default is `data`.')
parser.add_argument('--resume_filename',
                    type=str,
                    required=True,
                    nargs='+')
parser.add_argument('--weight',
                    type=float,
                    nargs='+',
                    default=None)
args = parser.parse_args()


if args.weight is None:
  print('`weight` is not specified. Automatically take the average of all logits.')
  args.weight = [1 / len(args.resume_filename) for _ in range(len(args.resume_filename))]
else:
  if len(weight) != len(resume_filename):
    raise ValueError('Length of `weight` and `resume_filename` must be equal.')
  if abs(sum(weight) - 1) > 1e-4:
    raise ValueError('Sum of `weight` must equal to 1.')

logits_lst = []
for file in args.resume_filename:
  if not os.path.isfile(os.path.join('logits', file)):
    raise OSError('File {} does not exist.'.format(file))
  logits_lst.append(torch.load(os.path.join('logits', file), map_location = lambda storage, loc: storage))

output = torch.sum(torch.cat([ts.unsqueeze(0) * w for ts, w in zip(logits_lst, args.weight)],0),0)

_, _, _, labels, idx_alltrain, _, idx_test = load_data(path=args.data_dir, percent=1.)

train_preds = torch.ge(output.float(), 0.5)[idx_alltrain]
train_labels = labels[idx_alltrain]

acc = accuracy(train_preds, train_labels)
f1 = f1_score(train_preds, train_labels)
print('Evaluate on the whole training set ({} nodes in total) : acc {:.4f}, f1 {:.4f}'.format(len(idx_alltrain), acc, f1))


blendname = 'blend-' + '-'.join([os.path.splitext(file)[0] for file in args.resume_filename]) + 'csv'
SUB_FILENAME = os.path.join('submission', blendname)
TMP_FILENAME = os.path.join('.temporary', blendname)

train_dict = {
  idx_alltrain[id].item():' '.join(list(map(str, row.nonzero().numpy().ravel().tolist()))) 
  for id, row in enumerate(train_preds)
}
train_annotated = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
train_annotated['Labels'] = train_annotated['Id'].apply(lambda x: train_dict[x])
train_annotated = train_annotated[['Id', 'Labels']]
if not os.path.isdir('.temporary'):
  os.makedirs('.temporary')
train_annotated.to_csv(TMP_FILENAME, index=False)

test_preds = torch.ge(output.float(), 0.5)[idx_test].cpu()
test_dict = {
  idx_test[id].item():' '.join(list(map(str, row.nonzero().numpy().ravel().tolist()))) 
  for id, row in enumerate(test_preds)
}
  
if not os.path.isdir('submission'):
  os.makedirs('submission')
submission = pd.read_csv(os.path.join(args.data_dir, 'sampleSubmission.csv'))
submission['Labels'] = submission['Id'].apply(lambda x: test_dict[x])
submission.to_csv(SUB_FILENAME, index=False)
