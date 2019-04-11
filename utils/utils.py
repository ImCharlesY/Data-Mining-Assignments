
import os
import json
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_data(percent, path="data"):
    # Load node ids
    idx_train = pd.read_csv(os.path.join(path, 'train.csv'))['Id'].values.ravel()
    idx_test = pd.read_csv(os.path.join(path, 'test.csv'))['Id'].values.ravel()
    idx_allnodes = np.hstack([idx_train, idx_test])
    split_idx = int(len(idx_train) * percent)
    idx_val = idx_train[split_idx:]
    idx_train = idx_train[:split_idx]

    # Build adjacency matrix
    edges_tbl = pd.read_csv(os.path.join(path, 'edges.csv'))
    rows, cols = edges_tbl['Source'].values, edges_tbl['Target'].values
    datas = edges_tbl['Weight'].values
    sp_adj = sp.csr_matrix((datas, (rows, cols)), shape=((len(idx_allnodes),)*2))
    # build symmetric adjacency matrix
    # sp_adj = sp_adj + sp_adj.T.multiply(sp_adj.T > sp_adj) - sp_adj.multiply(sp_adj.T > sp_adj)
    sp_adj = bilateral_normalize(sp_adj + sp.eye(sp_adj.shape[0]))

    # Load labels
    raw_labels = pd.read_csv(os.path.join(path, 'train.csv')).values
    id2labels = {row[0]:row[1:].tolist() for row in raw_labels}
    labels = np.zeros((len(idx_allnodes), raw_labels.shape[1] - 1))
    labels[list(id2labels.keys())] = np.asarray(list(id2labels.values()))

    # Load node features
    with open(os.path.join(path, 'node_info.json')) as f:
        node_features = json.load(f)
    features = {int(k):v for k,v in node_features.items()}
    
    # Get unique aff ids
    affs = set()
    for _, feat in features.items():
        for aff in feat['Affiliation']:
            affs.add(aff['AffiliationID'])

    # Convert features to sparse
    rows, cols, datas = [], [], []
    for id, feat in features.items():
        rows.append(id)
        cols.append(0)
        datas.append(feat['PaperCount'])
        rows.append(id)
        cols.append(1)
        datas.append(feat['CitationCountInAI'])
        for aff in feat['Affiliation']:
            rows.append(id)
            cols.append(aff['AffiliationID'] + 2)
            datas.append(aff['PaperCount'])
    sp_features = sp.csr_matrix((datas, (rows, cols)), shape=(len(idx_allnodes), max(affs) + 3),  dtype=np.float32)
    sp_features = unilateral_normalize(sp_features)

    features = torch.FloatTensor(np.array(sp_features.todense()))
    labels = torch.FloatTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(sp_adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def unilateral_normalize(mx):
    """Row-normalize"""
    rowsum = np.array(mx.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    mx = d_mat_inv.dot(mx)
    return mx


def bilateral_normalize(mx):
    """Symmetrically normalize."""
    rowsum = np.array(mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mx = d_mat_inv_sqrt.dot(mx).dot(d_mat_inv_sqrt)
    return mx


def accuracy(output, labels, threshold=0.5):
    preds = torch.ge(output.float(), threshold).float()
    correct = preds.eq(labels).float()
    correct = correct.sum()
    return correct / labels.numel()
    

def f1_score(output, labels, eps=1e-9, threshold=0.5):
    output = torch.ge(output.float(), threshold).float()
    labels = labels.float()
    true_positive = (output * labels).sum(dim=1)
    precision = true_positive.div(output.sum(dim=1).add(eps))
    recall = true_positive.div(labels.sum(dim=1).add(eps))
    return torch.mean((2 * precision * recall).div(precision + recall + eps))


def f1_loss(output, labels, eps=1e-9):
    output = output.float()
    labels = labels.float()
    true_positive = (output * labels).sum(dim=1)
    precision = true_positive.div(output.sum(dim=1).add(eps))
    recall = true_positive.div(labels.sum(dim=1).add(eps))
    f1 = torch.mean((2 * precision * recall).div(precision + recall + eps))
    return 1 - f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
