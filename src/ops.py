import numpy as np
import pandas as pd

def get_topk_index(matrix, topk=10):
    sorted_index = np.argsort(matrix)[:,::-1]
    return sorted_index[:, 1:topk+1]

def get_recommend_with_topk_user(purchase_matrix, topk_index, target_purchase_matrix=None):
    if target_purchase_matrix is None:
        target_purchase_matrix = purchase_matrix
    purchase_item_count = np.clip(purchase_matrix[topk_index] - target_purchase_matrix[:, np.newaxis, :], 0, 1)
    purchase_item_count = np.sum(purchase_item_count, axis=1)
    sorted_item_index = np.argsort(purchase_item_count)[:,::-1]
    sorted_item_count = np.take_along_axis(purchase_item_count, sorted_item_index, axis=1)
    return sorted_item_index, sorted_item_count

def get_recommend_with_topk_user_score(purchase_matrix, topk_index, similarity_score):
    topk_score = np.take_along_axis(similarity_score, topk_index, axis=1)
    topk_score = np.expand_dims(topk_score, axis=-1)
    purchase_item_matrix = purchase_matrix[topk_index]
    recormmend_item_socre = np.sum(purchase_item_matrix * topk_score, axis=1)
    sorted_item_index = np.argsort(recormmend_item_socre)[:,::-1]
    sorted_item_score = np.take_along_axis(recormmend_item_socre, sorted_item_index, axis=1)
    return sorted_item_index, sorted_item_score

def get_recommend_matrix(purchase_matrix, recommend_item_index, recommend_flag=2):
    for c_idx, col in enumerate(recommend_item_index):
        for row in col:
            purchase_matrix[c_idx][row] = recommend_flag
    return purchase_matrix
