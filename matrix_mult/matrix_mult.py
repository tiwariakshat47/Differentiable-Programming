# Soft matrix multiplication.
import autograd.numpy as np

def matrix_mult(A: np.ndarray, B: np.ndarray):
    # A is n x m
    # B is m x p
    # Ensure m's match for both matrices
    if A.shape[1] != B.shape[0]:
        print("matrix_mult(): invalid shape")
        return None
    
    n = A.shape[0]
    m = A.shape[1]
    p = B.shape[1]
    
    # Create result matrix with shape of A's rows and B's cols
    C = np.zeros((n, p))
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
                
    return C


def visualize_matrix(M):
    for row in M:
        for elem in row:
            print(elem, end=' ')
        print()

def visualize_matrix_mult(A, B):
    C = matrix_mult(A, B)
    visualize_matrix(A)
    print("*")
    visualize_matrix(B)
    print("=")
    visualize_matrix(C)



def main():
    A = np.array([[0,1,2,3],[0,1,2,3],[0,1,2,3]])
    B = np.array([[0,1,2],[0,1,2],[0,1,2],[0,1,2]])
    visualize_matrix_mult(A, B)

if __name__ == "__main__":
    main()