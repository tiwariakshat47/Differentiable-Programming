import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
from torchviz import make_dot
# import torchlens as tl
from tqdm import tqdm
import itertools

from soft_list import softindex, softget, softset, softswap
from loss_fns import mse_loss, corrcoef_loss
from plotting import plot_losses_wrt_offsets, plot_losses_offsets_wrt_epoch, plot_losses_sigmas



def bubble_sort(arr):
    # Classic bubble sort.
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


# A small variation with tmp variable (in-progress)
# def bubble_sort(arr):
#     for i in range(len(arr)):
#         for j in range(len(arr)-i-1):
#             tmp = a[j]

#             if arr[j] > arr[j+1]:
#                 arr[j], arr[j+1] = arr[j+1], arr[j]
#     return arr


def soft_bubble_sort(arr, offset, sigma): # iterations=1
    size = len(arr)
    arr = arr.clone()
    arr.requires_grad = True

    # for _ in range(iterations):  # TODO: why is iterations an argument?
    for i in range(size):
    # for i in range(round(size - offset.item())):
        j_range = torch.ceil(size - i - offset).int().item() # TODO: careful about this rounding
        for j in range(j_range):
            val1 = softget(arr, j, sigma)
            val2 = softget(arr, j + offset, sigma)

            if val1 > val2:
                arr = softswap(arr, j, j + offset, sigma)

    return arr


def soft_bubble_sort_noloop(arr, offset, sigma):
    size = len(arr)
    arr = arr.clone()
    arr.requires_grad = True
    arr.retain_grad()
    # print(f'arr.grad: {arr.grad}')
    # arr.register_hook(lambda grad: print(f'arr grad: {grad}'))

    i = 0
    val1 = softget(arr, i, sigma)
    val2 = softget(arr, i + offset, sigma)
    # print(offset.grad_fn)

    if val1 > val2:
        arr = softswap(arr, i, i + offset, sigma)

    return arr




def train():

    # Hyperparameters
    sigma = 0.28
    lr = 0.02
    num_epochs = 1
    start_offset = 2.0

    # Set requires_grad false so we don't modify ground truth array
    # array = torch.tensor([30.0, 1.0, 100.0, 5.0, 2.0, 75.0, 50.0, 10.0, 40.0, 60.0], requires_grad=False)  
    # array = torch.tensor([2.0, 6.0, 4.0, 1.0], requires_grad=False)
    array = torch.tensor([6.0, 2.0, 1.0], requires_grad=False)

    classical_sorted_array = torch.tensor(bubble_sort(array.clone().tolist()), dtype=torch.float32)

    # The single offset parameter which we're going to optimize
    offset_param = torch.tensor(start_offset, requires_grad=True)
    optimizer = torch.optim.Adam([offset_param], lr=lr)

    # For graphing
    losses = []
    soft_sorted_arrays = []
    offsets = []

    for epoch in range(num_epochs):
        print(f'\nEPOCH {epoch}')
        optimizer.zero_grad()

        soft_sorted_array = soft_bubble_sort(array, offset_param, sigma)
        loss = mse_loss(soft_sorted_array, classical_sorted_array)
        loss.backward()
        optimizer.step()
        print(f'soft_sorted_array: {soft_sorted_array}')
        print(f'loss: {loss}')
        print(f'offset: {offset_param}')
        print(f'offset grad: {offset_param.grad}')

        offsets.append(offset_param.item())
        losses.append(loss.item())
        soft_sorted_arrays.append(soft_sorted_array.detach().cpu().numpy())

        # if epoch % 50 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss.item()}, offset: {offset_param.item()}")

    print(f'\nEND')
    print(f"Softly sorted array: {soft_sorted_array}")
    print(f"Classically sorted array: {classical_sorted_array}")

    # plot_losses_offsets(losses, offsets)


def visualize_loss():
    # The optimal "offset" a swap in bubble_sort is 1. This function 
    # gives a visualization of the loss function so we can evaluate
    # whether the gradient is convex and if there is a minimum at 1.

    n_lists = 10
    list_size = 10
    # sigma = 0.35
    sigmas = torch.arange(0.28, 0.42, 0.06)
    offsets = torch.arange(0.5, 5, 0.1)
    loss_fn = mse_loss

    # arr = torch.tensor([30.0, 1.0, 100.0, 5.0, 2.0, 75.0, 50.0, 10.0, 40.0, 60.0], requires_grad=False)  
    # arr = torch.tensor([93., 74., 59., 84., 81.,  1., 20., 17., 22., 14.], requires_grad=False)
    # arr = torch.tensor([33., 74., 15., 76., 66., 60., 57., 99., 48., 12.], requires_grad=False)

    # Random arrays
    arrs = [torch.floor(torch.rand(list_size, requires_grad=False) * 100) for i in range(n_lists)]

    # All permutations of an array
    # arr = [float(n) for n in range(list_size)]
    # arrs = torch.tensor(list(itertools.permutations(arr)), requires_grad=False)

    sigma_losses = {}
    for sigma in sigmas:
        print(f'\nsigma: {sigma}')
        loss_lists = []
        for a in tqdm(arrs):
            losses = []
            for offset in offsets:
                soft_sorted = soft_bubble_sort(a, offset, sigma)
                classic_sorted = torch.tensor(bubble_sort(a.clone().tolist()))
                loss = loss_fn(soft_sorted, classic_sorted).item()
                losses.append(loss)
            loss_lists.append(losses)
        sigma_losses[sigma] = loss_lists

    mean_sigma_losses = {sigma: [sum(l) / len(l) for l in zip(*loss_lists)]
                         for sigma, loss_lists in sigma_losses.items()}
    
    plot_losses_sigmas(offsets, mean_sigma_losses, f"Averaged {loss_fn.__name__} Loss Gradient With {n_lists} Random Lists") # All Permutations {arr}

    # f"MSE Loss Gradient\narr={array.tolist()}")


def visualize_model():
    # Visualize the computation graph.

    list_size = 2
    # a = torch.tensor([4.0, 3.0, 2.0, 1.0], requires_grad=False)
    a = torch.floor(torch.rand(list_size, requires_grad=False) * 100)
    start_offset = 1.0
    sigma = 0.3

    offset_param = torch.tensor(start_offset, requires_grad=True)
    # soft_bubble_sort(a, offset_param, sigma)

    # Include the loss function in computation graph
    def f():
        pred = soft_bubble_sort(a, offset_param, sigma)
        target = torch.tensor(bubble_sort(pred.clone().tolist()), dtype=torch.float32)
        return mse_loss(pred, target)
    make_dot(f(), params={'offset_param': offset_param}).render('out', view=True) #, show_attrs=True, show_saved=True 


if __name__ == "__main__":
    train()
    # visualize_loss()
    # visualize_model()
