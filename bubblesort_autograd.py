# Trying out bubble sort with the autograd library.

import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam, sgd
import matplotlib.pyplot as plt
from tqdm import tqdm

from plotting import plot_losses_wrt_offsets, plot_losses_offsets_wrt_epoch, plot_losses_sigmas, plot_losses_wrt_epoch


def mse_loss(soft_sorted_array, classical_sorted_array):
    return np.mean((soft_sorted_array - classical_sorted_array) ** 2)

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


def softif(condition: float, true_branch: np.ndarray, false_branch: np.ndarray):
    return (condition * true_branch) - ((1.0 * condition) * false_branch)


def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


def bubble_sort_soft(arr, offset, sigma):
    for i in range(len(arr)):

        j_range = int(np.ceil(len(arr) - i - offset))
        for j in range(j_range):
            val1 = softget(arr, j, sigma)
            val2 = softget(arr, j + offset, sigma)

            if val1 > val2:
                arr = softswap(arr, j, j + offset, sigma)

    return arr


def train():
    sigma = 0.28
    lr = 0.01
    num_epochs = 40
    start_offset = 1.3

    arr = np.array([4.0, 3.0, 2.0, 1.0])

    classical_sorted_array = bubble_sort(arr.copy())

    losses = []
    offsets = []
    def objective(params, iter):
        offset = params['offset']
        soft_sorted_array = bubble_sort_soft(arr, offset, sigma)
        loss = mse_loss(soft_sorted_array, classical_sorted_array)
        print(f"loss: {loss._value}")
        losses.append(float(loss._value))
        offsets.append(float(offset._value))
        return loss

    def log_epoch(params, iter, gradient):
        print(f'\nepoch: {iter}')
        print(f"offset: {params['offset']}")
        # print(f'gradients: {gradient}')


    params = {"offset": start_offset}
    opt = sgd(
        grad(objective),
        params,
        num_iters=num_epochs,
        callback=log_epoch,
        step_size=lr
        )
    
    print(losses)
    print(offsets)
    plot_losses_offsets_wrt_epoch(losses, offsets)


def visualize_loss():

    list_size = 10
    n_lists = 10
    sigmas = np.arange(0.28, 0.42, 0.06)
    offsets = np.arange(0.5, 5, 0.1)
    loss_fn = mse_loss

    # Random arrays
    arrs = [np.floor(np.random.rand(list_size) * 100) for i in range(n_lists)]

    # Calculate loss for different values of offset and sigma
    sigma_losses = {}
    for sigma in sigmas:
        print(f'\nsigma: {sigma}')
        loss_lists = []
        for a in tqdm(arrs):
            losses = []
            for offset in offsets:
                soft_sorted = bubble_sort_soft(a, offset, sigma)
                classic_sorted = bubble_sort(a.copy())
                loss = loss_fn(soft_sorted, classic_sorted).item()
                losses.append(loss)
            loss_lists.append(losses)
        sigma_losses[sigma] = loss_lists

    # Get the mean loss across all the lists
    mean_sigma_losses = {sigma: [sum(l) / len(l) for l in zip(*loss_lists)]
                         for sigma, loss_lists in sigma_losses.items()}
    
    plot_losses_sigmas(offsets, mean_sigma_losses, f"Averaged {loss_fn.__name__} Loss Gradient With {n_lists} Random Lists") # All Permutations {arr}



if __name__ == "__main__":
    # train()
    visualize_loss()
