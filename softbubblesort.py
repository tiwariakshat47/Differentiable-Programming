# Trying out bubble sort with the autograd library.

import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam, sgd
import matplotlib.pyplot as plt
from tqdm import tqdm

from soft_list import softget, softset, softswap
from loss_fns import mse_loss, softrank_mse_loss
from plotting import plot_losses_wrt_param, plot_losses_offsets_wrt_epoch, plot_losses_sigmas, plot_losses_wrt_epoch


def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


def logistic(x):
    return 1 / (1 + np.exp(-x))


def softif(condition: float, true_branch: np.ndarray, false_branch: np.ndarray):
    return (condition * true_branch) - ((1.0 - condition) * false_branch)


def bubble_sort_soft(arr, offset, comparison_param, sigma):
    # comparison_param = 0.01

    for i in range(len(arr)):
        j_range = int(np.ceil(len(arr) - i - offset))
        for j in range(j_range):
            val1 = softget(arr, j, sigma)
            val2 = softget(arr, j + offset, sigma)

            # TODO: this is just a scaling param, should there be another branching param?
            # soft_compare = logistic((val1 - val2) / comparison_param)  
            # arr = softif(soft_compare, softswap(arr, j, j + offset, sigma), 1.0)

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


def visualize_offsets_sigmas_loss():

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
                soft_sorted = bubble_sort_soft(a, offset, 0.0001, sigma)
                classic_sorted = bubble_sort(a.copy())
                loss = loss_fn(soft_sorted, classic_sorted).item()
                losses.append(loss)
            loss_lists.append(losses)
        sigma_losses[sigma] = loss_lists

    # Get the mean loss across all the lists
    mean_sigma_losses = {sigma: [sum(l) / len(l) for l in zip(*loss_lists)]
                         for sigma, loss_lists in sigma_losses.items()}
    
    plot_losses_sigmas(offsets, mean_sigma_losses, f"Averaged {loss_fn.__name__} Loss Gradient With {n_lists} Random Lists") # All Permutations {arr}


def visualize_loss():

    list_size = 4
    n_lists = 10
    # sigmas = np.arange(0.28, 0.42, 0.06)
    sigma = 0.28
    # comparison_params = [0.01, 0.1, 1.0]#, 1.3, 1.7]
    comparison_params = np.arange(0.01, 10.0, 0.1)
    offsets = np.arange(0.5, 5, 0.1)
    loss_fn = softrank_mse_loss

    # Random arrays
    # arrs = [np.floor(np.random.rand(list_size) * 100) for i in range(n_lists)]
    # arr = np.floor(np.random.rand(list_size) * 100)
    arr = [3,2,1]
    print(f'arr: {arr}')

    losses = []
    for comparison_param in comparison_params:
        soft_sorted = bubble_sort_soft(arr, offset=1, comparison_param=comparison_param, sigma=sigma)
        classic_sorted = bubble_sort(arr.copy())
        loss = loss_fn(soft_sorted, classic_sorted).item()
        losses.append(loss)

    
    # plot_losses_sigmas(offsets, mean_sigma_losses, f"Averaged {loss_fn.__name__} Loss Gradient With {n_lists} Random Lists") # All Permutations {arr}
    plot_losses_wrt_param(losses, comparison_params, "Loss wrt Comparison Parameter")



if __name__ == "__main__":
    # train()
    visualize_offsets_sigmas_loss()
    # visualize_loss()

