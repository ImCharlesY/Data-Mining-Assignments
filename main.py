#!/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Charles
'''

from __future__ import absolute_import

import os
import sys
import time
import logging
from utils.option import args
from models.trainer import Trainer

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)
LOG_FILENAME = os.path.join(args.output_dir, '{}.log'.format(time.strftime('%Y%m%d_%H%M%S', time.localtime())))
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    handlers = [
                        logging.FileHandler(LOG_FILENAME, mode='w'),
                        logging.StreamHandler(sys.stdout)
                    ])

def main():
  model = Trainer(args)
  model.train()
  model.resume()
  model.evaluate()
  model.submit()

if __name__ == "__main__":
  main()   
