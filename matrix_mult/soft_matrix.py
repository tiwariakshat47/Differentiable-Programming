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
    distances = cdist(all_points, target_points)
    
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
    # visualize_matrix(weights * M)
    return np.sum(weights * M)


def visualize_matrix(M):
    for row in M:
        for elem in row:
            print(f"{elem:.5f}", end=' ')
        print()

if __name__ == "__main__":
    # M = [[0,1,2,3],[0,1,2,3],[0,1,2,3]]
    M = [[1,2,3,40],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
    M = np.array(M)
    print("M:")
    visualize_matrix(M)

    print("\n----- Test softget -----")
    sigma = 0.4
    i, j = 0, 1
    print(f'hardget({i}, {j}): {M[i,j]}')
    print(f'softget({i}, {j}, sigma={sigma}): {softget_matrix(M, i, j, sigma)}')
