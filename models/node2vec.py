#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import os
import torch
import logging
import numpy as np

def node2vec(graph, embedding_dir, sym=0):
  if sym == 0 or sym == 2:
    return _node2vec(graph, embedding_dir, 0)
  elif sym == 1 or sym == 3:
    return _node2vec(graph, embedding_dir, 1)
  elif sym in [4,5,6]:
    return torch.cat([_node2vec(graph, embedding_dir, 0), _node2vec(graph, embedding_dir, 1)],1)
  else:
    raise ValueError('`sym` only accepts value in [0,4].')


def _node2vec(graph, embedding_dir, sym=0):
    
  EMBEDDING_FILENAME = os.path.join(embedding_dir, 'embeddings{}.emb'.format(sym))
  EMBEDDING_MODEL_FILENAME = os.path.join(embedding_dir, 'embeddings{}.model'.format(sym))

  logging.info('Try to load embedding file from `{}`...'.format(EMBEDDING_FILENAME))
  if not os.path.isfile(EMBEDDING_FILENAME):
    from gensim.models import Word2Vec
    from node2vec import Node2Vec

    logging.warning('Cannot find embedding file. Try to load Node2Vec model from `{}`...'.format(EMBEDDING_MODEL_FILENAME))
    if os.path.isfile(EMBEDDING_MODEL_FILENAME):
      model = Word2Vec.load(EMBEDDING_MODEL_FILENAME)
    else:
      logging.warning('Cannot find model file. Start to train a new model. IT WILL TAKE A LONG TIME...')
      # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
      node2vec = Node2Vec(graph.reverse() if sym else graph, dimensions=256, walk_length=30, num_walks=200, workers=4)
      # Embed nodes
      model = node2vec.fit(window=10, min_count=1, batch_words=4)
      model.save(EMBEDDING_MODEL_FILENAME)
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)
    del model

  from gensim.models import KeyedVectors
  wv = KeyedVectors.load_word2vec_format(EMBEDDING_FILENAME, binary=False)
  features = np.vstack([wv[str(nodeid)] for nodeid in range(graph.number_of_nodes())])
  features = torch.FloatTensor(features)
  return features
