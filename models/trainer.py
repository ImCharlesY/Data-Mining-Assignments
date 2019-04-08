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

import os
import sys
import json
import random
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
from torchnet import meter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from models.bert import BertForQA
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils.evaluate import evaluate
from utils.option import parse_config
from utils.container import SquadExample, InputFeatures, RawResult
from utils.preprocessing import read_squad_examples, convert_examples_to_features
from utils.postprocessing import write_predictions, convert_features_loss_to_examples_loss


class Trainer(object):

  def __init__(self, args):
    self.args = args
    self.device = torch.device("cuda", self.args.cuda[0]) if torch.cuda.is_available() else torch.device("cpu")
    self.n_gpu = torch.cuda.device_count()
    logging.info("device: {} n_gpu: {} parallelization: {}".format(self.device, self.n_gpu, len(self.args.cuda)>1))

    random.seed(self.args.seed)
    np.random.seed(self.args.seed)
    torch.manual_seed(self.args.seed)
    if self.n_gpu > 0:
      torch.cuda.manual_seed_all(self.args.seed)

    self.train_dataloader = None
    self.eval_dataloader = None
    self.test_dataloader = None

    self.build_model()
    logging.info(parse_config(self.args.__dict__))


  def build_model(self):

    self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=True)
    
    # Prepare model
    self.model = BertForQA.from_pretrained(self.args.bert_model,
                cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE))
    self.model.to(self.device)
    if self.n_gpu > 1 and len(self.args.cuda) > 1:
      self.model = torch.nn.DataParallel(self.model, device_ids=self.args.cuda)


  def prepare_optimizer(self):
    # Prepare optimizer
    param_optimizer = list(self.model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]

    self.optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=self.args.learning_rate,
                         warmup=self.args.warmup_proportion,
                         t_total=self.num_train_optimization_steps)


  def load_training_data(self):
    if self.train_dataloader is not None:
      return

    logging.info("Load training data...")
    train_examples = read_squad_examples(
      input_file=self.args.train_file, is_training=True)
    self.num_train_optimization_steps = int(len(train_examples) / self.args.train_batch_size) * self.args.num_train_epochs

    cached_train_features_file = self.args.train_file+'_{0}_{1}_{2}_{3}'.format(
      list(filter(None, self.args.bert_model.split('/'))).pop(), str(self.args.max_seq_length), str(self.args.doc_stride), str(self.args.max_query_length))
    train_features = None
    try:
      with open(cached_train_features_file, "rb") as reader:
        train_features = pickle.load(reader)
    except:
      train_features = convert_examples_to_features(
        examples=train_examples,
        tokenizer=self.tokenizer,
        max_seq_length=self.args.max_seq_length,
        doc_stride=self.args.doc_stride,
        max_query_length=self.args.max_query_length,
        is_training=True)
      logging.info("  Saving train features into cached file %s", cached_train_features_file)
      with open(cached_train_features_file, "wb") as writer:
        pickle.dump(train_features, writer)
    logging.info("***** Running training *****")
    logging.info("  Num orig examples = %d", len(train_examples))
    logging.info("  Num split examples = %d", len(train_features))
    logging.info("  Batch size = %d", self.args.train_batch_size)
    logging.info("  Num steps = %d", self.num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_example_index,
                               all_start_positions, all_end_positions)

    train_sampler = RandomSampler(train_data)
    self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)
    self.train_examples = train_examples
    self.train_features = train_features


  def load_evaluating_data(self):
    if self.eval_dataloader is not None:
      return

    logging.info("Load evaluating data...")
    eval_examples = read_squad_examples(
      input_file=self.args.eval_file, is_training=True)

    cached_eval_features_file = self.args.eval_file+'_{0}_{1}_{2}_{3}'.format(
      list(filter(None, self.args.bert_model.split('/'))).pop(), str(self.args.max_seq_length), str(self.args.doc_stride), str(self.args.max_query_length))
    eval_features = None
    try:
      with open(cached_eval_features_file, "rb") as reader:
        eval_features = pickle.load(reader)
    except:
      eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=self.tokenizer,
        max_seq_length=self.args.max_seq_length,
        doc_stride=self.args.doc_stride,
        max_query_length=self.args.max_query_length,
        is_training=True)
      logging.info("  Saving eval features into cached file %s", cached_eval_features_file)
      with open(cached_eval_features_file, "wb") as writer:
        pickle.dump(eval_features, writer)
    logging.info("***** Running training *****")
    logging.info("  Num orig examples = %d", len(eval_examples))
    logging.info("  Num split examples = %d", len(eval_features))
    logging.info("  Batch size = %d", self.args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_example_index,
                               all_start_positions, all_end_positions)

    eval_sampler = SequentialSampler(eval_data)
    self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
    self.eval_examples = eval_examples
    self.eval_features = eval_features


  def load_testing_data(self):
    if self.test_dataloader is not None:
      return

    logging.info("Load test data...")
    test_examples = read_squad_examples(
      input_file=self.args.predict_file, is_training=True, fake=0)
    # Load Ans1
    test_features = convert_examples_to_features(
      examples=test_examples,
      tokenizer=self.tokenizer,
      max_seq_length=self.args.max_seq_length,
      doc_stride=self.args.doc_stride,
      max_query_length=self.args.max_query_length,
      is_training=True)
    # Load Ans2
    test_features_2 = convert_examples_to_features(
      examples=read_squad_examples(input_file=self.args.predict_file, is_training=True, fake=1),
      tokenizer=self.tokenizer,
      max_seq_length=self.args.max_seq_length,
      doc_stride=self.args.doc_stride,
      max_query_length=self.args.max_query_length,
      is_training=True)    

    logging.info("***** Running predictions *****")
    logging.info("  Num orig examples = %d", len(test_examples))
    logging.info("  Num split examples = %d", len(test_features))
    logging.info("  Batch size = %d", self.args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    # Ans1
    ans1_start_positions = torch.tensor([f.start_position for f in test_features], dtype=torch.long)
    ans1_end_positions = torch.tensor([f.end_position for f in test_features], dtype=torch.long)
    # Ans2
    ans2_start_positions = torch.tensor([f.start_position for f in test_features_2], dtype=torch.long)
    ans2_end_positions = torch.tensor([f.end_position for f in test_features_2], dtype=torch.long)    

    all_start_positions = torch.cat([ans1_start_positions.view(-1,1), ans2_start_positions.view(-1,1)], 1)
    all_end_positions = torch.cat([ans1_end_positions.view(-1,1), ans2_end_positions.view(-1,1)], 1)    

    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_example_index, 
                              all_start_positions, all_end_positions)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    self.test_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.predict_batch_size)
    self.test_examples = test_examples
    self.test_features = test_features


  def train(self):
    if os.path.exists(self.args.output_dir) and os.path.exists(os.path.join(self.args.output_dir, WEIGHTS_NAME)):
      raise ValueError("Output directory {} already exists and is not empty.".format(self.args.output_dir))
    self.load_training_data()
    self.prepare_optimizer()
    self.model.train()
    loss_meter = meter.AverageValueMeter()
    inner_step = 0
    for epoch in trange(int(self.args.num_train_epochs), desc="Epoch", ascii=True, ncols=100):
      loss_meter.reset()
      for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration", ascii=True, ncols=100)):
        batch.pop(3) # Drop example index
        if len(self.args.cuda) == 1:
          batch = tuple(t.to(self.device) for t in batch) # multi-gpu does scattering it-self
        loss = self.model(*batch)[0]
        if self.n_gpu > 1 and len(self.args.cuda) > 1:
          loss = loss.mean() # mean() to average on multi-gpu.
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        loss_meter.add(loss.item())
        inner_step += 1
    self.save()


  def save(self):
    if not os.path.exists(self.args.output_dir):
      os.makedirs(self.args.output_dir)
    # Save a trained model and the associated configuration
    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
    output_model_file = os.path.join(self.args.output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(self.args.output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
      f.write(model_to_save.config.to_json_string())


  def resume(self):
    if self.args.resume_dir is None:
      raise ValueError("resume_dir is not specified.")
    # Load a trained model and config that you have fine-tuned
    output_config_file = os.path.join(self.args.resume_dir, CONFIG_NAME)
    output_model_file = os.path.join(self.args.resume_dir, WEIGHTS_NAME)
    config = BertConfig(output_config_file)
    self.model = BertForQA(config)
    self.model.load_state_dict(torch.load(output_model_file, map_location = lambda storage, loc: storage))
    self.model.to(self.device)


  def evaluate(self):
    self.load_evaluating_data()
    self.model.eval()
    all_results = []
    logging.info("Start evaluating")
    for input_ids, segment_ids, input_mask, example_indices, _, _ in tqdm(self.eval_dataloader, desc="Evaluating", ascii=True, ncols=100):
      input_ids = input_ids.to(self.device)
      segment_ids = segment_ids.to(self.device)
      input_mask = input_mask.to(self.device)
      with torch.no_grad():
        batch_start_logits, batch_end_logits = self.model(input_ids, segment_ids, input_mask)
      for i, example_index in enumerate(example_indices):
        start_logits = batch_start_logits[i].detach().cpu().tolist()
        end_logits = batch_end_logits[i].detach().cpu().tolist()
        eval_feature = self.eval_features[example_index.item()]
        unique_id = int(eval_feature.unique_id)
        all_results.append(RawResult(unique_id=unique_id,
                                     start_logits=start_logits,
                                     end_logits=end_logits))
    if not os.path.exists(self.args.output_dir):
      os.makedirs(self.args.output_dir)
    output_prediction_file = os.path.join(self.args.output_dir, "predictions_eval.json")
    write_predictions(self.eval_examples, self.eval_features, all_results,
                      self.args.n_best_size, self.args.max_answer_length,
                      output_prediction_file, self.args.verbose_logging)
    with open(os.path.splitext(self.args.eval_file)[0] + '.json') as dataset_file:
      dataset = json.load(dataset_file)['data']
    with open(output_prediction_file) as prediction_file:
      predictions = json.load(prediction_file)
    logging.info(json.dumps(evaluate(dataset, predictions)))


  def submit(self):
    self.load_testing_data()
    self.model.eval()
    all_results = []
    all_loss1 = torch.empty(0).to(self.device)
    all_loss2 = torch.empty(0).to(self.device)
    # all_start_probas = torch.empty(0)
    # all_end_probas = torch.empty(0)
    # all_start_pos1 = torch.empty(0).long()
    # all_end_pos1 = torch.empty(0).long()
    # all_start_pos2 = torch.empty(0).long()
    # all_end_pos2 = torch.empty(0).long()
    logging.info("Start generating submission file")
    for input_ids, segment_ids, input_mask, example_indices, start_positions, end_positions in tqdm(self.test_dataloader, desc="Predicting", ascii=True, ncols=100):
      input_ids, segment_ids, input_mask = input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device)
      start_positions, end_positions = start_positions.to(self.device), end_positions.to(self.device)
      with torch.no_grad():
        loss1, batch_start_logits, batch_end_logits = self.model(input_ids, segment_ids, input_mask, start_positions[:,0], end_positions[:,0], mean=False)
        loss2 = self.model(input_ids, segment_ids, input_mask, start_positions[:,1], end_positions[:,1], mean=False)[0]
        all_loss1 = torch.cat([all_loss1, loss1], 0)
        all_loss2 = torch.cat([all_loss2, loss2], 0)
        # batch_start_logits, batch_end_logits = self.model(input_ids, segment_ids, input_mask)
        # batch_start_log_probas = -torch.log(torch.softmax(batch_start_logits, dim=1))
        # batch_end_log_probas = -torch.log(torch.softmax(batch_end_logits, dim=1))
        # all_start_probas = torch.cat([all_start_probas, batch_start_log_probas.cpu()], 0)
        # all_end_probas = torch.cat([all_end_probas, batch_end_log_probas.cpu()], 0)
        # all_start_pos1 = torch.cat([all_start_pos1, start_positions[:,0].cpu().view(-1)], 0)
        # all_end_pos1 = torch.cat([all_end_pos1, end_positions[:,0].cpu().view(-1)], 0)
        # all_start_pos2 = torch.cat([all_start_pos2, start_positions[:,1].cpu().view(-1)], 0)
        # all_end_pos2 = torch.cat([all_end_pos2, end_positions[:,1].cpu().view(-1)], 0)

      for i, example_index in enumerate(example_indices):
        start_logits = batch_start_logits[i].detach().cpu().tolist()
        end_logits = batch_end_logits[i].detach().cpu().tolist()
        eval_feature = self.test_features[example_index.item()]
        unique_id = int(eval_feature.unique_id)
        all_results.append(RawResult(unique_id=unique_id,
                                     start_logits=start_logits,
                                     end_logits=end_logits))
    all_loss1 = convert_features_loss_to_examples_loss(self.test_examples, self.test_features, all_loss1.cpu().numpy().tolist())
    all_loss2 = convert_features_loss_to_examples_loss(self.test_examples, self.test_features, all_loss2.cpu().numpy().tolist())
    # all_loss1 = convert_features_loss_to_examples_loss(self.test_examples, self.test_features, 
    #             all_start_probas.numpy().tolist(), all_end_probas.numpy().tolist(), all_start_pos1.numpy().tolist(), all_end_pos1.numpy().tolist())
    # all_loss2 = convert_features_loss_to_examples_loss(self.test_examples, self.test_features, 
    #             all_start_probas.numpy().tolist(), all_end_probas.numpy().tolist(), all_start_pos2.numpy().tolist(), all_end_pos2.numpy().tolist()) 

    ans_cat = np.asarray([all_loss1, all_loss2]).argmin(0) + 1
    ans_id = np.arange(1, len(self.test_examples) + 1)
    ans_tbl = pd.DataFrame(data={'Id':ans_id,'Category':ans_cat})
    logging.info("The submission file has {} prediction rows".format(len(ans_tbl)))
    if not os.path.exists(self.args.output_dir):
      os.makedirs(self.args.output_dir)
    ans_tbl.to_csv(os.path.join(self.args.output_dir, 'submission.csv'), index=False)

    output_prediction_file = os.path.join(self.args.output_dir, "predictions_test.json")
    write_predictions(self.test_examples, self.test_features, all_results,
                      self.args.n_best_size, self.args.max_answer_length,
                      output_prediction_file, self.args.verbose_logging)
