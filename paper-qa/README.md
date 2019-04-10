# PaperQA

This repository is for the first assignment in Big Data Mining lesson (EE448). We are required to compete in a [Kaggle in-class competition](https://kaggle.com/c/ee448-paperqa) by training a natural language processing (NLP) model to handle the reading comprehension task. The model we used is __Bidirectional Encoder Representations from Transformers (BERT)__ proposed by Google AI Research ([Paper](https://arxiv.org/abs/1810.04805)). 

## Environment

### Requirements

#### General
- python (verified on 3.6.7)

#### Python Packages
- torch (verified on 1.0.1)
- pandas (verified on 0.24.2)
- numpy (verified on 1.16.2)
- tqdm (verified on 4.31.1)
- torchnet (verified on 0.0.4)
- pytorch_pretrained_bert (verified on 0.6.1)

### Setup with anaconda or pip

Setup a Python virtual environment (optional):

```
conda create -n m_env python=3.6.7
source activate m_env
```

Install the requirements:

``` 
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```


## Detailed Documentation

|  Directory or file name  |               description                   |
| ------------------------ |:-------------------------------------------:|
| ./data/ | directory to the dataset (train.csv, test.csv, QAmapping.md) |
| ./models/ | contains the BERT adapted to SQuAD, and the PyTorch trainer | 
| ./utils/ | contains all functional scripts including argparser, pre-processing and post-processing |
| ./dataset_mining.ipynb | notebook to show how we pre-process the dataset |
| ./main.py | top script |

The `main.py` script in root directory is the top script. __You can run it with default settings to re-produce our results. FYI, you can specify the `--cuda` option to determine which and how many GPUs should be used.__

For example:

```
python main.py --cuda 0          # use only GPU0
python main.py --cuda 0 1 2 3    # use 4 GPUs from ID0 to ID3
```

### main.py

```
usage: main.py [-h] [--bert_model BERT_MODEL] --output_dir OUTPUT_DIR
               [--resume_dir RESUME_DIR] [--train_file TRAIN_FILE]
               [--eval_file EVAL_FILE] [--predict_file PREDICT_FILE]
               [--max_seq_length MAX_SEQ_LENGTH] [--doc_stride DOC_STRIDE]
               [--max_query_length MAX_QUERY_LENGTH]
               [--train_batch_size TRAIN_BATCH_SIZE]
               [--eval_batch_size EVAL_BATCH_SIZE]
               [--predict_batch_size PREDICT_BATCH_SIZE]
               [--learning_rate LEARNING_RATE]
               [--num_train_epochs NUM_TRAIN_EPOCHS]
               [--warmup_proportion WARMUP_PROPORTION]
               [--n_best_size N_BEST_SIZE]
               [--max_answer_length MAX_ANSWER_LENGTH] [--verbose_logging]
               [--cuda CUDA [CUDA ...]] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --bert_model BERT_MODEL
                        Bert pre-trained model selected in the list: bert-
                        base-uncased, bert-large-uncased, bert-base-cased,
                        bert-large-cased, bert-base-multilingual-uncased,
                        bert-base-multilingual-cased, bert-base-chinese.
  --output_dir OUTPUT_DIR
                        The output directory where the model checkpoints and
                        predictions will be written.
  --resume_dir RESUME_DIR
                        The resume directory.
  --train_file TRAIN_FILE
                        SQuAD json for training. E.g., train-v1.1.json
  --eval_file EVAL_FILE
                        SQuAD json for evaluating. E.g., dev-v1.1.json
  --predict_file PREDICT_FILE
                        SQuAD json for predictions. E.g., dev-v1.1.json or
                        test-v1.1.json
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        WordPiece tokenization. Sequences longer than this
                        will be truncated, and sequences shorter than this
                        will be padded.
  --doc_stride DOC_STRIDE
                        When splitting up a long document into chunks, how
                        much stride to take between chunks.
  --max_query_length MAX_QUERY_LENGTH
                        The maximum number of tokens for the question.
                        Questions longer than this will be truncated to this
                        length.
  --train_batch_size TRAIN_BATCH_SIZE
                        Total batch size for training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Total batch size for evaluating.
  --predict_batch_size PREDICT_BATCH_SIZE
                        Total batch size for predictions.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --warmup_proportion WARMUP_PROPORTION
                        Proportion of training to perform linear learning rate
                        warmup for. E.g., 0.1 = 10% of training.
  --n_best_size N_BEST_SIZE
                        The total number of n-best predictions to generate in
                        the nbest_predictions.json output file.
  --max_answer_length MAX_ANSWER_LENGTH
                        The maximum length of an answer that can be generated.
                        This is needed because the start and end predictions
                        are not conditioned on one another.
  --verbose_logging     If true, all of the warnings related to data
                        processing will be printed. A number of warnings are
                        expected for a normal SQuAD evaluation.
  --cuda CUDA [CUDA ...]
                        The ids of CUDAs to be used if available. If multiple
                        values are given, the procedure will be optimized by
                        multi-gpu parallelization.
  --seed SEED           random seed for initialization.
```

## Acknowledgments

This code uses the PyTorch-Implementation of the BERT from [huggingface's repository](https://github.com/huggingface/pytorch-pretrained-BERT). Thanks for your great work.