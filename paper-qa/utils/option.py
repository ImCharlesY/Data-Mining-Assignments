#!/usr/bin/env
# -*- coding: utf-8 -*-

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2019, Charles
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import

import numpy as np
import argparse


def parse_config(args):
  strout = '\n'
  strout += '-' * 70 + '\n'
  strout += ' {:^20} | {:^40} '.format('Name', 'Value') + '\n'
  strout += '-' * 70 + '\n'
  for k, v in args.items():
    if v is None:
      v = ''
    elif isinstance(v, list) or isinstance(v, np.ndarray):
      v = '[' + ', '.join(str(e) for e in v) + ']'
    strout += ' {:^20} | {:^40} '.format(k, v) + '\n'
  strout += '-' * 70
  strout += '\n'
  return strout


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--output_dir", default="experiments/test", type=str,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--resume_dir", default="experiments/test", type=str, 
                    help="The resume directory.")

## Other parameters
parser.add_argument("--train_file", default="data/all.csv", type=str, help="SQuAD json for training. E.g., train-v1.1.json")
parser.add_argument("--eval_file", default="data/eval.csv", type=str, help="SQuAD json for evaluating. E.g., dev-v1.1.json")
parser.add_argument("--predict_file", default="data/test.csv", type=str,
                    help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
parser.add_argument("--max_seq_length", default=384, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=128, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--max_query_length", default=64, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                         "be truncated to this length.")
parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for evaluating.")
parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=10.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                         "of training.")
parser.add_argument("--n_best_size", default=20, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json "
                         "output file.")
parser.add_argument("--max_answer_length", default=160, type=int,
                    help="The maximum length of an answer that can be generated. This is needed because the start "
                         "and end predictions are not conditioned on one another.")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")
parser.add_argument("--cuda", default=[0], type=int, nargs='+',
                    help="The ids of CUDAs to be used if available.\n"
                         "If multiple values are given, the procedure will be optimized by multi-gpu parallelization.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
args = parser.parse_args()
