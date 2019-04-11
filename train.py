from __future__ import absolute_import

import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchcrf import CRF

from models.gcn import GCN
from utils.utils import load_data, accuracy, f1_score, f1_loss

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--train_percent', default=0.2, type=float,
                    help='The percent of dataset to be used as training set.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--submission', type=str, default='tmp.csv',
                    help='Submission filename.')

parser.add_argument('--cuda', default=0, type=int, 
                    help='The ids of CUDA to be used if available.')
parser.add_argument('--loss_fun', default='f1', type=str, choices=['f1','bce','crf'],
                    help='Loss function.')
parser.add_argument('--optim', type=str, default='adam', choices=['adam','sgd','adagrad','rmsprop'],
                    help='Optimizer to be used.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
device = torch.device('cuda', args.cuda) if torch.cuda.is_available() else torch.device('cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.train_percent)
print('\n'.join([
      'Num of nodes: {}'.format(adj.shape[0]),
      'Feature dimension: {}'.format(features.shape[1]),
      'Num of training set: {}'.format(len(idx_train)),
      'Num of validation set: {}'.format(len(idx_val)),
      'Num of testing set: {}'.format(len(idx_test)),
      ]))

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nlabel=labels.size(1),
            dropout=args.dropout,
            crf=args.loss_fun=='crf').to(device)

optim_handle = {'adam':optim.Adam, 'sgd':optim.SGD, 'adagrad':optim.Adagrad, 'rmsprop':optim.RMSprop}
optimizer = optim_handle[args.optim](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fun_handle = {'f1':f1_loss, 'bce':nn.BCELoss(), 'crf':CRF(2, batch_first=True).to(device)}
criterion = loss_fun_handle[args.loss_fun]

features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    if args.loss_fun == 'crf':
      loss_train = criterion(output[idx_train], labels[idx_train].long())
    else:
      loss_train = criterion(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if args.loss_fun == 'crf':
      prediction = features.new_tensor(criterion.decode(output[idx_train]))
    else:
      prediction = output[idx_train]
    acc_train = accuracy(prediction, labels[idx_train])
    f1_train = f1_score(prediction, labels[idx_train])      

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    if args.loss_fun == 'crf':
      loss_val = criterion(output[idx_val], labels[idx_val].long())
    else:
      loss_val = criterion(output[idx_val], labels[idx_val])
    if args.loss_fun == 'crf':
      prediction = features.new_tensor(criterion.decode(output[idx_val]))
    else:
      prediction = output[idx_val]
    acc_val = accuracy(prediction, labels[idx_val])
    f1_val = f1_score(prediction, labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'f1_train: {:.4f}'.format(f1_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'f1_val: {:.4f}'.format(f1_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    if args.loss_fun == 'crf':
      output = features.new_tensor(criterion.decode(output))

    test_preds = torch.ge(output.float(), 0.5)[idx_test].cpu()
    test_dict = {
      idx_test[id].item():' '.join(list(map(str, row.nonzero().numpy().ravel().tolist()))) 
      for id, row in enumerate(test_preds)}
      
    if not os.path.isdir('submission'):
        os.makedirs('submission')
    submission = pd.read_csv(os.path.join('data', 'sampleSubmission.csv'))
    submission['Labels'] = submission['Id'].apply(lambda x: test_dict[x])
    submission.to_csv(os.path.join('submission', args.submission), index=False)

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
