import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

def compute_cosine_similarity(matrix_a, matrix_b=None):
    if matrix_b is not None:
        return cosine_similarity(matrix_a, matrix_b)
    else:
        return cosine_similarity(matrix_a, matrix_a)

def compute_jaccard_similarity(matrix_a, matrix_b=None):
    if matrix_b is not None:
        n_sample_a = matrix_a.shape[0]
        n_sample_b = matrix_b.shape[0]
        similarity_matrix = np.zeros((n_sample_a, n_sample_b))
        for i in range(n_sample_a):
            for j in range(n_sample_b):
                similarity_matrix[i][j] = jaccard_score(matrix_a[i], matrix_b[j])
                similarity_matrix[j][i] = similarity_matrix[i][j]
        return similarity_matrix

    else:
        n_sample = matrix.shape[0]
        similarity_matrix = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            for j in range(i+1, n_sample):
                similarity_matrix[i][j] = jaccard_score(matrix[i], matrix[j])
        return np.eye(n_sample, k=0) + similarity_matrix + similarity_matrix.T