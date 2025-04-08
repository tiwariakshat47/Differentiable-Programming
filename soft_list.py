import numpy as np

def softindex(size: int, index: int | float, sigma: float):
    # Gaussian indexing.
    # size: size of the array/distribution to be generated
    # index: center of distribution
    # sigma: "width" of the distribution
    indices = np.arange(size)
    distances = indices - index
    exponent = -0.5 * (distances / sigma) ** 2
    weights = np.exp(exponent)
    weights = weights / weights.sum()
    return weights

def softget(arr: np.ndarray, index: int | float, sigma: float):
    # Get the soft value of a tensor using soft indexing.
    # Calculate the gaussian index distribution for the arr
    dist = softindex(len(arr), index, sigma)

    # Softget
    return np.dot(dist, arr)


def softset(arr, index, value, sigma):
    # linear interpolation?
    #example: use complement to not scale down: ensures that the other elements retain their original values, weighted appropriately
    # origin: arr = [1, 2, 3], index dist: w = [0.2, 0.5, 0.3] if val to be set is 10
    # then do --> i[0] --> new_arr[0]=(1−0.2)⋅1+0.2⋅10=0.8⋅1+2.0= 2.8
    # i[1] --> new_arr[1]=(1−0.5)⋅2+0.5⋅10=0.5⋅2+5.0= 6.0
    # i[2] --> new_arr[2]=(1−0.3)⋅3+0.3⋅10=0.7⋅3+3.0= 5.1
    # index_distribution = index_distribution.clone().detach()
    index_distribution = softindex(len(arr), index, sigma)
    return arr * (1 - index_distribution) + index_distribution * value


def softswap(arr, index_1, index_2, sigma):
    value1 = softget(arr, index_1, sigma)
    value2 = softget(arr, index_2, sigma)

    arr = softset(arr, index_1, value2, sigma)
    arr = softset(arr, index_2, value1, sigma)

    return arr


# TODO: write tests here