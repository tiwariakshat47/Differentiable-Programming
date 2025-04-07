import numpy as np
from scipy.spatial.distance import cdist


def get_weight_matrix(M, i, j, sigma=0.2):
    rows, cols = M.shape

    # Unravel matrix into a list of coordinates
    row_indices, col_indices = np.indices((rows, cols))
    all_points = np.stack((row_indices.flatten(), col_indices.flatten()), axis=1)

    # Convert point (i,j) to an array of identical points
    target_points = np.full(all_points.shape, np.array([i, j]))

    # Calculate the Euclidean distances between all points and target point (i,j)
    # O(n^2)  (I think, two for loops)
    distances = cdist(all_points, target_points, metric="euclidean")
    
    # Find the minimum distance for each point to any of the target points
    min_distances = np.min(distances, axis=1)

    # Apply gaussian to the min distances
    exponent = -0.5 * (min_distances / sigma) ** 2
    weights = np.exp(exponent)
    weights = weights / weights.sum()

    # Reshape the weights back into the original matrix shape
    weights = weights.reshape((rows, cols))

    return weights

def softget_matrix(M, i, j, sigma):
    weights = get_weight_matrix(M, i, j, sigma)
    return np.sum(weights * M)

def softset_matrix(M, i, j, value, sigma):
    weights = get_weight_matrix(M, i, j, sigma)
    return (M * (1 - weights)) + (weights * value)

def visualize_matrix(M):
    # TODO: align columns
    print()
    for row in M:
        for elem in row:
            print(f"    {elem:.5f}", end=' ')
        print()

if __name__ == "__main__":
    print("----- Soft Matrix Testing -----")
    # M = [[0,1,2,3],[0,1,2,3],[0,1,2,3]]
    M = [[1,2,3,40],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
    sigma = 0.4
    M = np.array(M)
    print("M:")
    print(M)

    print("\n----- Test softget -----")
    i, j = 0, 1
    print(f'hardget({i}, {j}): {M[i,j]}')
    print(f'softget({i}, {j}, sigma={sigma}): {softget_matrix(M, i, j, sigma)}')

    print("\n----- Test softset -----")
    i, j = 0, 1
    sigma = 0.4
    value = 100
    M_hard = M.copy()
    M_hard[i,j] = 100
    print(f'hardset({i}, {j}, {value}): ')
    print(M_hard, end='\n\n')
    print(f'softset({i}, {j}, {value}, {sigma}):')
    print(softset_matrix(M, i, j, value, sigma), end='\n\n')
