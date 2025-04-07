"""Imports
"""
import torch
import autograd.numpy as np

# ------------------------------- #
# Classical Matrix Multiplication #
# ------------------------------- #

"""Check if tensors A and B are multiplicable
"""
def canMultiply(A, B):
    # Let A be mxn and B be pxq (rows x cols)
    # A and B are multiplicable if n = p
    # Number of cols in A must be equal to number of rows in B
    return (A.shape[1] == B.shape[0])

"""Returns whether or not M is a scalar
"""
def isScalar(M):
    # Check if M is a single value tensor, i.e. a scalar
    return M.numel() == 1

"""Returns the number of rows in tensor M
"""
def numRows(M):
    return M.shape[0]

"""Returns number of columns in tensor M
"""
def numCols(M):
    return M.shape[1]

"""Performs scalar multiplication of S and M, where S is the scalar tensor
"""
def scalarMult(S, M):
    # scalar = S.item()
    # P = np.zeros((numRows(M), numCols(M)))
    # print(f"P = {P}")
    # for row in range(numRows(M)):
    #     for col in range(numCols(M)):
    #         P[row][col] = scalar * M[row][col]
    # return P
    return S.item() * M


"""
    Classical matrix multiplication.
    Input: two matrices, A and B
    Output: a single product matrix, A*B
"""
def classical_matrix_mult(A, B):
    # Edge case
    # if not A or not B:
    #     print("Error, A and/or B is empty.\nA = {A}\nB = {B}")
    #     return

    # Optimize cases where A and/or B are scalars
    if isScalar(A) and isScalar(B): # A and B are both single values
        return A.item() * B.item()
    elif isScalar(A):               # Just A is a single value
        print(f"A = {A.item()}")
        return scalarMult(A, B)
    elif isScalar(B):               # Just B is a single value
        print(f"B = {B}")
        return scalarMult(B, A)
    
    print(f"A shape = {A.shape}")
    print(f"B shape = {B.shape}")
    # Verify that A and B are multiplicable before proceeding
    if not canMultiply(A, B):
        print(f"Error, {numCols(A)} != {numRows(B)}")
        return

    # Initialize the product matrix P, which will be # rows in A by # cols in B
    Prod = torch.zeros(numRows(A), numCols(B))

    # Perform line-by-line multiplication of matrices A and B
    for i in range(numRows(A)):
        for j in range(numCols(B)):
            for k in range(numCols(A)):
                Prod[i][j] += A[i][k] * B[k][j]
    return Prod


"""Classical matrix multiplication tests
"""
torch.set_printoptions(linewidth=100)
# A = torch.tensor([[1, 2, 3],
#                   [4, 5, 6]])
A = torch.tensor([2, 3, 4])
B = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])
AB = classical_matrix_mult(A, B)
print(f"A = {A}")
print(f"B = {B}")
print(f"AB = {AB}")





