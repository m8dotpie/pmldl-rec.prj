"""
Model evaluations utility functions
"""
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim, Tensor

from torch_sparse import SparseTensor, matmul

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj


def bpr_loss(
    users_emb_final,
    users_emb_0,
    pos_items_emb_final,
    pos_items_emb_0,
    neg_items_emb_final,
    neg_items_emb_0,
    lambda_val,
):
    reg_loss = lambda_val * (
        users_emb_0.norm(2).pow(2)
        + pos_items_emb_0.norm(2).pow(2)
        + neg_items_emb_0.norm(2).pow(2)
    )  # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)

    return -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss


# helper function to get N_u
def get_user_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


# computes recall@K and precision@K
def RP(groundTruth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor(
        [len(groundTruth[i]) for i in range(len(groundTruth))]
    )
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


# computes NDCG@K
def NDCG(groundTruth, r, k):
    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1.0 / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1.0 / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.0
    return torch.mean(ndcg).item()


# wrapper function to get evaluation metrics
def get_metrics(model, edge_index, exclude_edge_indices, k):
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    rating = torch.matmul(user_embedding, item_embedding.T)
    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        rating[exclude_users, exclude_items] = -(1 << 10)

    _, top_K_items = torch.topk(rating, k=k)

    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype("float"))

    recall, precision = RP(test_user_pos_items_list, r, k)
    ndcg = NDCG(test_user_pos_items_list, r, k)

    return recall, precision, ndcg


# wrapper function to evaluate model
def evaluation(
    model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val
):
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index
    )
    edges = structured_negative_sampling(edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = (
        users_emb_final[user_indices],
        users_emb_0[user_indices],
    )
    pos_items_emb_final, pos_items_emb_0 = (
        items_emb_final[pos_item_indices],
        items_emb_0[pos_item_indices],
    )
    neg_items_emb_final, neg_items_emb_0 = (
        items_emb_final[neg_item_indices],
        items_emb_0[neg_item_indices],
    )

    loss = bpr_loss(
        users_emb_final,
        users_emb_0,
        pos_items_emb_final,
        pos_items_emb_0,
        neg_items_emb_final,
        neg_items_emb_0,
        lambda_val,
    ).item()

    recall, precision, ndcg = get_metrics(model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg
