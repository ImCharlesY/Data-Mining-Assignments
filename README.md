# Node-Classification

This repository is for the second assignment in Big Data Mining lesson (EE448). We are required to compete in a [Kaggle in-class competition](https://www.kaggle.com/c/ee448-2019-node-classification) by training a machine learning model to handle a multi-label node classification based on citation network.

## Environment

### Requirements

#### General
- python (verified on 3.6.7)

#### Python Packages
- torch (verified on 1.0.1)
- torchvision (verified on 0.3.0)
- pandas (verified on 0.24.2)
- numpy (verified on 1.16.2)
- scipy (verified on 1.2.1)
- scikit-learn (verified on 0.21.2)
- node2vec (verified on 0.3.0)
- tensorflow-gpu (verified on 1.13.1)

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
| ./data/ | directory to the dataset (train.csv, test.csv, node_info.json) |
| ./models/ | contains all models we implement | 
| ./utils/ | contains all functional scripts including argparser, pre-processing and evaluation metric |
| ./dataset_mining.ipynb | notebook to show how we pre-process the dataset |
| ./main.py | top script |
| ./blend.py | script for ensemble learning (not used) |

The `main.py` script in root directory is the top script.

## How To Use

__In order to re-produce our online results, you can run the following command:__

```
python main.py --submission submission.csv --embedding 0 --sym 4 --net ngcn --epochs 50 --train_percent 1.0
```

__FYI, you can specify the `--cuda` option to determine which GPU should be used.__

For more available arguments, you can run `python main.py -h` to get the following help messages:

```
usage: main.py [-h] [--seed SEED] [--submission SUBMISSION]
               [--data_dir DATA_DIR] [--multinet] [--embedding {0,1,2}]
               [--sym {0,1,2,3,4,5,6}] [--pca PCA]
               [--net {gcn,ngcn,gccn,gat,hat,bp}] [--ghid GHID] [--fhid FHID]
               [--number_of_layers NUMBER_OF_LAYERS] [--dropout DROPOUT]
               [--cuda CUDA] [--train_percent TRAIN_PERCENT] [--epochs EPOCHS]
               [--early-stopping EARLY_STOPPING] [--loss_fun {f1,bce,hamming}]
               [--optim {adam,sgd,adagrad,rmsprop}] [--lr LR]
               [--weight_decay WEIGHT_DECAY]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random seed. Default is 0.
  --submission SUBMISSION
                        Submission filename.
  --data_dir DATA_DIR   Dataset directory. Default is `data`.
  --multinet            Specify to apply multinet architecture. (`embedding`,
                        `sym` will be overwritten to be 0)
  --embedding {0,1,2}   0 - Only Node2Vec embedding; 1 - Only original
                        features; 2 - Both (concatenate). Default is 0.
  --sym {0,1,2,3,4,5,6}
                        0 - Original adjacency matrix; 1 - Transpose adjacency
                        matrix; 2 - Convert adjacency matirx to symmetric
                        (embedding 0); 3 - Convert adjacency matirx to
                        symmetric (embedding 1); 4 - Convert adjacency matrix
                        to symmetric (concatenate embedding0 and embedding1);
                        5 - Original adjacency matrix (concatenate embedding0
                        and embedding1); 6 - Transpose adjacency matrix
                        (concatenate embedding0 and embedding1);
  --pca PCA             The percentage of the amount of variance that needs to
                        be explained in PCA. Default is 0 (do not use).
  --net {gcn,ngcn,gccn,gat,hat,bp}
                        Network architecture. Default is `gcn`.
  --ghid GHID           Number of hidden units in graph neural network.
                        Ignored when use ngcn. Default is 128.
  --fhid FHID           Number of output dimension of the subnet in multinet.
                        Default is 256.
  --number_of_layers NUMBER_OF_LAYERS
                        Number of hidden layers in the preprocessing fcn.
                        Ignored when `embedding` is not equal to 3. Default is
                        3.
  --dropout DROPOUT     Dropout rate (1 - keep probability). Default is 0.5.
  --cuda CUDA           The ids of CUDA to be used if available. Default is 0.
  --train_percent TRAIN_PERCENT
                        The percent of dataset to be used as training set.
                        Default is 0.2.
  --epochs EPOCHS       Number of epochs to train. Default is 500.
  --early-stopping EARLY_STOPPING
                        Number of early stopping rounds. Default is 20.
  --loss_fun {f1,bce,hamming}
                        Loss function. Default is `f1`.
  --optim {adam,sgd,adagrad,rmsprop}
                        Optimizer to be used. Default is `adam`.
  --lr LR               Initial learning rate. Default is 0.005.
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 loss on parameters). Default is 5e-4.
```

## Acknowledgments

This code we used is based on the following repositories. Thanks for their work.

- PyTorch-Implementation of GCN from [tkipf's repository](https://github.com/tkipf/pygcn)

- PyTorch-Implementation of GAT from [PetarV-'s repository](https://github.com/PetarV-/GAT)

- Example of usage of node2vec from [eliorc's repository](https://github.com/eliorc/node2vec#usage)

