# Implementations of soft list functions as described in The Elements of 
# Differentiable Programming by Blondel and Roulet.

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def softindex(size: int, index: int | float, sigma: float):
    # Gaussian indexing.
    # size: size of the array/distribution to be generated
    # index: center of distribution
    # sigma: "width" of the distribution
    indices = torch.arange(size)
    distances = indices - index
    exponent = -0.5 * (distances / sigma) ** 2
    weights = torch.exp(exponent)
    weights = weights / weights.sum()
    return weights

def softget(arr: torch.Tensor, index: int | float | torch.Tensor, sigma: float):
    # Get the soft value of a tensor using soft indexing.
    # Calculate the gaussian index distribution for the arr
    dist = softindex(arr.size(0), index, sigma)

    # Softget
    return torch.dot(dist, arr)


def softset(arr, index, value, sigma):
    # linear interpolation?
    #example: use complement to not scale down: ensures that the other elements retain their original values, weighted appropriately
    # origin: arr = [1, 2, 3], index dist: w = [0.2, 0.5, 0.3] if val to be set is 10
    # then do --> i[0] --> new_arr[0]=(1−0.2)⋅1+0.2⋅10=0.8⋅1+2.0= 2.8
    # i[1] --> new_arr[1]=(1−0.5)⋅2+0.5⋅10=0.5⋅2+5.0= 6.0
    # i[2] --> new_arr[2]=(1−0.3)⋅3+0.3⋅10=0.7⋅3+3.0= 5.1
    # index_distribution = index_distribution.clone().detach()
    index_distribution = softindex(arr.size(0), index, sigma)
    return arr * (1 - index_distribution) + index_distribution * value


def softswap(arr, index_1, index_2, sigma):
    value1 = softget(arr, index_1, sigma)
    value2 = softget(arr, index_2, sigma)

    arr = softset(arr, index_1, value2, sigma)
    arr = softset(arr, index_2, value1, sigma)

    return arr


if __name__ == "__main__":
    print("Testing soft_list functions.")

    a = torch.tensor([0.0, 3.0, 10.0, 2.0], requires_grad=False)
    print(f"Testing with array: {a}")

    print("\n--- softindex() ---")
    print(f"softindex(a, index=0, sigma=1.0): {softindex(a.size(0), 0, 1.0)}")
    print(f"softindex(a, index=0.5, sigma=1.0): {softindex(a.size(0), 0.5, 1.0)}")
    print(f"softindex(a, index=1, sigma=0.4): {softindex(a.size(0), 1, 0.4)}")
    print(f"softindex(a, index=0.5, sigma=0.4): {softindex(a.size(0), 0.5, 0.4)}")

    print("\n--- softget() ---")
    print(f"softget(a, index=0, sigma=1.0): {softget(a, 0, 1.0)}")
    print(f"softget(a, index=0.5, sigma=1.0): {softget(a, 0.5, 1.0)}")
    print(f"softget(a, index=1, sigma=0.4): {softget(a, 1, 0.4)}")
    print(f"softget(a, index=0.5, sigma=0.4): {softget(a, 0.5, 0.4)}")

    sigma = 0.3
    indices = torch.arange(0, a.size(0)-1, 0.1)
    soft_steps = [softget(a, i, sigma) for i in indices]
    plt.figure(figsize=(10, 5))
    plt.plot(indices,soft_steps)
    plt.xlabel("index")
    plt.ylabel("softget(index)")
    plt.title(f"array: {a}, sigma: {sigma}")
    plt.show()

    print("\n--- softset() ---")
    print(f"softset(a, index=2, value=0.0, sigma=0.5): {a} -> {softset(a, 2, 0.0, 0.5)}")

    # TODO: softswap

