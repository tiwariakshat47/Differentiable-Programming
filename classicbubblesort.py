import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from soft_list import softindex, softget, softset, softswap

# 
#classic bubble sort
def bubble_sort(arr):
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


#permutation distance or swap distance instead
def compute_loss(soft_sorted_array, classical_sorted_array):
    return torch.mean((soft_sorted_array - classical_sorted_array) ** 2)


def soft_bubble_sort(arr, offset, sigma, iterations=1):
    size = len(arr)
    arr = arr.clone()
    arr.requires_grad = True

    for _ in range(iterations):
        # i_range = round(torch.sub(torch.Tensor([size]), offset).item())
        for i in range(round(size - offset.item())):  # TODO: careful about this rounding
            val1 = softget(arr, i, sigma)
            val2 = softget(arr, i + offset, sigma)

            if val1 > val2:
                arr = softswap(arr, i, i + offset, sigma)

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


# requires_grad false so we don't modify ground truth array
# array = torch.tensor([30.0, 1.0, 100.0, 5.0, 2.0, 75.0, 50.0, 10.0, 40.0, 60.0], requires_grad=False)  
array = torch.tensor([2.0, 1.0], requires_grad=False)

sigma = 0.4
# distributions = apply_gaussian_to_array(array, sigma)
# print(distributions)

classical_sorted_array = torch.tensor(bubble_sort(array.clone().tolist()), dtype=torch.float32)

offset_param = torch.tensor(0.1, requires_grad=True)
optimizer = torch.optim.SGD([offset_param], lr=0.1)

losses = []
soft_sorted_arrays = []
offsets = []

num_epochs = 15

for epoch in range(num_epochs):
    print(f'\nEPOCH {epoch}')
    optimizer.zero_grad()

    torch.autograd.set_detect_anomaly(True)
    soft_sorted_array = soft_bubble_sort_noloop(array, offset_param, sigma)
    loss = compute_loss(soft_sorted_array, classical_sorted_array)
    # loss.register_hook(lambda grad: print(f'loss grad: {grad}'))
    loss.backward()
    print(f'soft_sorted_array: {soft_sorted_array}')
    print(f'loss: {loss}')
    print(f'offset: {offset_param}')
    print(f'offset grad: {offset_param.grad}')
    optimizer.step()

    offsets.append(offset_param.item())
    losses.append(loss.item())
    soft_sorted_arrays.append(soft_sorted_array.detach().cpu().numpy())

    # if epoch % 50 == 0:
    #     print(f"Epoch {epoch}, Loss: {loss.item()}, offset: {offset_param.item()}")

print(f'\nEND')
print(f"Softly sorted array: {soft_sorted_array}")
print(f"Classically sorted array: {classical_sorted_array}")

# plt.figure(figsize=(10, 5))
# plt.plot([[loss, offset] for loss, offset in zip(losses, offsets)])
# plt.plot(loss)
fig, ax = plt.subplots(figsize = (10, 5))
ax2 = plt.twinx()
ax.plot(losses, color = 'g')
ax2.plot(offsets, color = 'r')
plt.title("Loss and Offset Curve")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss", color = 'g')
ax2.set_ylabel("Offset", color = 'r')
plt.show()
