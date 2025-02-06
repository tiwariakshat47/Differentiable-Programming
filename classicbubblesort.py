import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#!pip install -q transformers


#classic bubble sort
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


# def bubble_sort(arr):
#     for i in range(len(arr)):
#         for j in range(len(arr)-i-1):
#             tmp = a[j]

#             if arr[j] > arr[j+1]:
#                 arr[j], arr[j+1] = arr[j+1], arr[j]
#     return arr


def apply_gaussian_to_array(array, sigma=1.0):
    size = len(array)
    distributions = []
    for i in range(size):
        indices = np.arange(size)
        distances = indices - i
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        weights /= weights.sum()
        distributions.append(torch.tensor(weights, dtype=torch.float32))
    return distributions

#Creating 4 functions razvan suggested: softindex (done above), softget, softswap, softset

def softget(arr, index_distribution):
    return torch.dot(index_distribution, arr)



def softset(arr, index_distribution, value):
    # linear interpolation?
    #example: use complement to not scale down: ensures that the other elements retain their original values, weighted appropriately
    # origin: arr = [1, 2, 3], index dist: w = [0.2, 0.5, 0.3] if val to be set is 10
    # then do --> i[0] --> new_arr[0]=(1−0.2)⋅1+0.2⋅10=0.8⋅1+2.0= 2.8
    # i[1] --> new_arr[1]=(1−0.5)⋅2+0.5⋅10=0.5⋅2+5.0= 6.0
    # i[2] --> new_arr[2]=(1−0.3)⋅3+0.3⋅10=0.7⋅3+3.0= 5.1
    # index_distribution = index_distribution.clone().detach()
    return arr * (1 - index_distribution) + index_distribution * value



def softswap(arr, index_dist1, index_dist2):
    arr = arr.clone() 
    value1 = softget(arr, index_dist1)
    value2 = softget(arr, index_dist2)

    new_arr = softset(arr, index_dist1, value2)
    new_arr = softset(new_arr, index_dist2, value1)

    return new_arr


#permutation distance or swap distance instead

def compute_loss(soft_sorted_array, classical_sorted_array):
    return torch.mean((soft_sorted_array - classical_sorted_array) ** 2)




def soft_bubble_sort(arr, distributions, y, iterations=1):
    size = len(arr)
    arr = arr.clone()
    arr.requires_grad = True

    for _ in range(iterations):
        # print(y)
        # print(torch.Tensor([size]))
        # print(torch.sub(torch.Tensor([size]), y))
        i_range = round(torch.sub(torch.Tensor([size]), y).item())
        for i in range(i_range):
            dist1 = distributions[i]
            dist2_i = torch.add(torch.Tensor([i]), y)
            # print(y.grad)
            dist2 = distributions[round(dist2_i.item())]
            val1 = softget(arr, dist1)
            val2 = softget(arr, dist2)
            # print(arr.grad)

            if val1 > val2:
                arr = softswap(arr, dist1, dist2)
    
    # print(arr.grad_fn)

    return arr




array = torch.tensor([30.0, 1.0, 100.0, 5.0, 2.0, 75.0, 50.0, 10.0, 40.0, 60.0], requires_grad=False)  # keep this as false so we don't modify actual array
# [1, 2, 5, 10, 30, 40, 50, 60, 75, 100]
# [4, 0, 9, ]

# [0, 0, 0, 0, 1, 0, 0, 0, 0]
# [1, 0, 0, 0, 0, 0, 0, 0, 0]

sigma = 0.3
distributions = apply_gaussian_to_array(array, sigma)

classical_sorted_array = torch.tensor(bubble_sort(array.clone().tolist()), dtype=torch.float32)

dist_param = torch.tensor(2.0, requires_grad=True)
optimizer = torch.optim.SGD([dist_param], lr=0.3)

losses = []
soft_sorted_arrays = []

num_epochs = 4

for epoch in range(num_epochs):
    optimizer.zero_grad()

    soft_sorted_array = soft_bubble_sort(array, distributions, y=dist_param, iterations=5)
    loss = compute_loss(soft_sorted_array, classical_sorted_array)
    # print(loss)
    loss.register_hook(lambda grad: print(f'loss grad: {grad}'))
    loss.backward()
    print(f'soft_sorted_array: {soft_sorted_array.grad}')
    print(f'soft_sorted_array grad_fn: {soft_sorted_array.grad_fn}')
    # print(f'loss: {loss}')
    # print(f'loss grad: {loss.grad}')
    optimizer.step()
    # print(f'dist_param: {dist_param}')
    # print(f'dist_param grad: {dist_param.grad}')
    losses.append(loss.item())
    soft_sorted_arrays.append(soft_sorted_array.detach().cpu().numpy())
    # dist_param.data = torch.clamp(dist_param.data, min=1.0, max=2.0)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, y: {dist_param.item()}")

print(f"Softly sorted array: {soft_sorted_array}")
print(f"Classically sorted array: {classical_sorted_array}")

# plt.figure(figsize=(10, 5))
# plt.plot(losses)
# plt.title("Loss Curve")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()
