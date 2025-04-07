# Working on Differentiable Matrix Multiplication using autograd

"""Imports
"""
import torch
import numpy as np
from scipy.spatial.distance import cdist

# ------------------------------------ #
# Differentiable Matrix Multiplication #
# ------------------------------------ #
def visualize_matrix(M):
    for row in M:
        for elem in row:
            print(elem, end=' ')
        print()




if __name__ == "__main__":
    A = np.array([[0,1,2,3],[0,1,2,3],[0,1,2,3]])
    B = np.array([[0,1,2],[0,1,2],[0,1,2],[0,1,2]])
    visualize_matrix(A)
    visualize_matrix(B)