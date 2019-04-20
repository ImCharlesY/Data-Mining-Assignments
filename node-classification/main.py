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
import logging
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from models.bp import BP
from models.gcn import GCN
from models.ngcn import NGCN
from models.gat import GAT, HAT
from models.multinet import MultiNet
from models.node2vec import node2vec
from sklearn.decomposition import PCA

from utils.parse import parse_args
from utils.utils import load_data, parse_config
from utils.measure import accuracy, f1_score, f1_loss, hamming_loss


args = parse_args()

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
if args.submission.lower() == 'timestamp':
  args.submission = 'submission_{}.csv'.format(timestamp)

LOG_FILENAME = os.path.join('logs', '{}.log'.format(timestamp))
OLT_FILENAME = os.path.join('logits', '{}.pth'.format(timestamp))
SUB_FILENAME = os.path.join('submission', args.submission)
TMP_FILENAME = os.path.join('.temporary', args.submission)

if not os.path.isdir('logs'):
  os.makedirs('logs')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)8s] --  %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='w'),
        logging.StreamHandler(sys.stdout)
    ])

device = torch.device('cuda', args.cuda) if torch.cuda.is_available() else torch.device('cpu')

# Reset random state for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.multinet:
  args.sym = 0
  args.embedding = 0

# Load dataset
graph, adj, features, labels, idx_train, idx_val, idx_test = load_data(path=args.data_dir, percent=args.train_percent, sym=args.sym)
embedding = node2vec(graph, args.data_dir, args.sym)
if args.multinet:
  embedding1 = node2vec(graph, args.data_dir, 1)
del graph
gc.collect()
if args.pca > 0:
  aff_features = PCA(args.pca, whiten = True).fit_transform(features.numpy())
  features = torch.FloatTensor(aff_features)
if args.embedding == 0:
  del features
  features = embedding.to(device)
  msg = 'Uses only the embedding features (extracting from Node2Vec model).'
  if args.multinet:
    features1 = embedding1.to(device)
    msg = '`multinet` specified, load embeddings for both adj and adj.T.'
elif args.embedding == 1:
  del embedding
  features = features.to(device)
  msg = 'Uses only the original features.'
elif args.embedding == 2:
  features = torch.cat([features, embedding], 1).to(device)
  msg = 'Uses both the original features and the embedding features (extracting from Node2Vec model) '\
  + '-- simply concatenating.'
logging.info(msg)

adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

logging.info('\n'+'\n'.join([
      'Num of nodes: {}'.format(adj.shape[0]),
      'Feature dimension: {}'.format(features.shape[1] if args.embedding != 3 else features[1].shape[1]),
      '' if not args.multinet else 'Feature1 dimension: {}'.format(features1.shape[1]),
      'Num of training set: {}'.format(len(idx_train)),
      'Num of validation set: {}'.format(len(idx_val)),
      'Num of testing set: {}'.format(len(idx_test)),
      ]))


# Model, optimizer, loss function
modules = {'bp':BP,'gcn':GCN,'ngcn':NGCN,'gat':GAT}
if args.multinet:
  features = [features, features1]
  adj = [adj, adj.t()]
  model = MultiNet(nfeat=features[0].shape[1],
                   nhid=args.ghid,
                   nmid=args.fhid,
                   nlabel=labels.size(1),
                   dropout=args.dropout,
                   nfeat1=features[1].shape[1],
                   module=modules[args.net]).to(device)
elif args.net == 'hat':
  adj = [adj, adj.t()]
  model = HAT(nfeat=features.shape[1],
              nhid=args.ghid,
              nlabel=labels.size(1),
              nsemantic=len(adj),
              dropout=args.dropout).to(device)
else:
  model = modules[args.net](nfeat=features.shape[1],
                            nhid=args.fhid if args.net == 'bp' else args.ghid,
                            nlabel=labels.size(1),
                            dropout=args.dropout).to(device)

logging.info('\nNetwork architecture:\n{}'.format(str(model)))

optim_handle = {'adam':optim.Adam, 'sgd':optim.SGD, 'adagrad':optim.Adagrad, 'rmsprop':optim.RMSprop}
optimizer = optim_handle[args.optim](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fun_handle = {'f1':f1_loss, 'bce':nn.BCELoss(), 'hamming':hamming_loss}
criterion = loss_fun_handle[args.loss_fun]
logging.info(parse_config(args.__dict__) + '\n')


def train(epoch):
  t = time.time()

  model.train()
  optimizer.zero_grad()
  output = model(features, adj)
  loss_train = criterion(output[idx_train], labels[idx_train])
  loss_train.backward()
  optimizer.step()

  acc_train = accuracy(output[idx_train], labels[idx_train])
  f1_train = f1_score(output[idx_train], labels[idx_train])      

  model.eval()
  output = model(features, adj)

  loss_val = criterion(output[idx_val], labels[idx_val])
  acc_val = accuracy(output[idx_val], labels[idx_val])
  f1_val = f1_score(output[idx_val], labels[idx_val])
  loss_val = loss_val if not torch.isnan(loss_val) else criterion(output[idx_train], labels[idx_train])
  acc_val = acc_val if not torch.isnan(acc_val) else accuracy(output[idx_train], labels[idx_train])
  f1_val = f1_val if not torch.isnan(f1_val) else f1_score(output[idx_train], labels[idx_train])
  logging.info(' '.join([
    'Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'f1_train: {:.4f}'.format(f1_train.item()),
        'loss_val: {:.4f}'.format(loss_val.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),
        'f1_val: {:.4f}'.format(f1_val.item()),
        'time: {:.4f}s'.format(time.time() - t),
  ]))
  return f1_val


def evaluate():
  model.eval()
  output = model(features, adj)

  idx_alltrain = torch.cat([idx_train, idx_val], 0)
  train_preds = torch.ge(output.float(), 0.5)[idx_alltrain].cpu()
  train_labels = labels[idx_alltrain].cpu()

  acc = accuracy(train_preds, train_labels)
  f1 = f1_score(train_preds, train_labels)

  logging.info('Evaluate on the whole training set ({} nodes in total) : acc {:.4f}, f1 {:.4f}'.format(len(idx_alltrain), acc, f1))
  
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
  logging.info('Prediction on the whole training set is saved to `{}`.'.format(TMP_FILENAME))  


def submit():
  model.eval()
  output = model(features, adj)

  if not os.path.isdir('logits'):
    os.makedirs('logits')
  logging.info('Save the output logits tensor to {}'.format(OLT_FILENAME))
  torch.save(output, OLT_FILENAME)

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
  logging.info('Submission table is saved to `{}`.'.format(SUB_FILENAME))


try:
  # Train model
  t_total = time.time()
  best_metric = 0
  no_improvement = 0
  for epoch in range(args.epochs):
    metric = train(epoch)
    if metric < best_metric:
      no_improvement += 1
      if no_improvement == args.early_stopping:
        logging.info('Early stopping..')
        break
    else:
      no_improvement = 0
      best_metric = metric
      best_model = model.state_dict()
  logging.info('Optimization Finished! Best metric on validation set: {:.4f}.'.format(best_metric))
  logging.info('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

  # Evaluate
  model.load_state_dict(best_model)
  evaluate()
  submit()
except:
  print('')
  for file in [LOG_FILENAME, TMP_FILENAME, SUB_FILENAME]:
    if os.path.isfile(file):
      print('Remove {}'.format(file))
      os.system('rm {}'.format(file))
  import traceback
  traceback.print_exc()
