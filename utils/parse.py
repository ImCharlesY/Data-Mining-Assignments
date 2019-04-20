#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed',
                      type=int,
                      default=0, 
            help='Random seed. Default is 0.')

  parser.add_argument('--submission',
                      type=str,
                      default='timestamp',
            help='Submission filename.')

  parser.add_argument('--data_dir', 
                      type=str,
                      default='data',
            help='Dataset directory. Default is `data`.')

  parser.add_argument('--multinet',
                      action='store_true',
              help='Specify to apply multinet architecture. (`embedding`, `sym` will be overwritten to be 0)')

  parser.add_argument('--embedding',
                      type=int,
                      default=0,
                      choices=[0,1,2],
            help='0 - Only Node2Vec embedding; 1 - Only original features; 2 - Both (concatenate). Default is 0.')

  parser.add_argument('--sym',
                      type=int,
                      default=0,
                      choices=[0,1,2,3,4,5,6],
            help='0 - Original adjacency matrix; 1 - Transpose adjacency matrix;' 
            + ' 2 - Convert adjacency matirx to symmetric (embedding 0);'
            + ' 3 - Convert adjacency matirx to symmetric (embedding 1);'
            + ' 4 - Convert adjacency matrix to symmetric (concatenate embedding0 and embedding1);'
            + ' 5 - Original adjacency matrix (concatenate embedding0 and embedding1);'
            + ' 6 - Transpose adjacency matrix (concatenate embedding0 and embedding1);')

  parser.add_argument('--pca',
                      type=float,
                      default=0,
            help='The percentage of the amount of variance that needs to be explained in PCA. Default is 0 (do not use).')

  parser.add_argument('--net', 
                      type=str,
                      default='gcn',
                      choices=['gcn', 'ngcn', 'gccn', 'gat', 'hat', 'bp'],
            help='Network architecture. Default is `gcn`.')

  parser.add_argument('--ghid',
                      type=int,
                      default=128,
            help='Number of hidden units in graph neural network. Ignored when use ngcn. Default is 128.')

  parser.add_argument('--fhid',
                      type=int,
                      default=256,
            help='Number of output dimension of the subnet in multinet. Default is 256.')

  parser.add_argument('--number_of_layers',
                      type=int,
                      default=3,
            help='Number of hidden layers in the preprocessing fcn. Ignored when `embedding` is not equal to 3. Default is 3.')

  parser.add_argument('--dropout',
                      type=float,
                      default=0.5,
            help='Dropout rate (1 - keep probability). Default is 0.5.')

  parser.add_argument('--cuda',
                      type=int, 
                      default=0,
            help='The ids of CUDA to be used if available. Default is 0.')

  parser.add_argument('--train_percent',
                      type=float,
                      default=0.2,
            help='The percent of dataset to be used as training set. Default is 0.2.')

  parser.add_argument('--epochs',
                      type=int,
                      default=500,
            help='Number of epochs to train. Default is 500.')

  parser.add_argument('--early-stopping',
                      type=int,
                      default=20,
            help = 'Number of early stopping rounds. Default is 20.')

  parser.add_argument('--loss_fun',
                      type=str,
                      default='f1',
                      choices=['f1','bce','hamming'],
            help='Loss function. Default is `f1`.')

  parser.add_argument('--optim',
                      type=str,
                      default='adam',
                      choices=['adam','sgd','adagrad','rmsprop'],
            help='Optimizer to be used. Default is `adam`.')

  parser.add_argument('--lr',
                      type=float,
                      default=0.005,
            help='Initial learning rate. Default is 0.005.')

  parser.add_argument('--weight_decay',
                      type=float,
                      default=5e-4,
            help='Weight decay (L2 loss on parameters). Default is 5e-4.')

  return parser.parse_args()