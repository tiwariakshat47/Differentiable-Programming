import numpy as np
from scipy.stats import norm  # For Gaussian CDF


def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]

#GCF function using bounds a, b
def GCF(a, b, hyperparam):
    """
    Gaussian Cumulative Function (GCF) between bounds a and b.

    Args:
        a (float): Lower bound.
        b (float): Upper bound.
        temperature (float): Controls the sharpness of the function.

    Returns:
        float: GCF weight for the given range.
    """
    return norm.cdf(b, scale=hyperparam) - norm.cdf(a, scale=hyperparam)

#using math from andy
def soft_index(array_length, hyperparam=1.0):
    """
    Generate soft indices as a probability distribution.

    Args:
        array_length (int): Length of the array.
        soft_position (float): The soft index position.
        temperature (float): Controls the sharpness of the weights.

    Returns:
        np.ndarray: Normalized weights for the soft index.
    """
    weights = np.zeros(array_length)
    for i in range(array_length):
        if i == 0:  
            weights[i] = GCF(-np.inf, i + 0.5, hyperparam)
        elif i == array_length - 1:  
            weights[i] = GCF(i - 0.5, np.inf, hyperparam)
        else:  
            weights[i] = GCF(i - 0.5, i + 0.5, hyperparam)
    print(weights/weights.sum())
    return weights / weights.sum() 


def soft_get(array, soft_index_weights):
    """
    Perform a soft get operation using weights.

    Args:
        array (np.ndarray): The input array.
        soft_index_weights (np.ndarray): Probability distribution over the array indices.

    Returns:
        float: The softly accessed value.
    """
    return np.dot(soft_index_weights, array)  

#not sure if this is right
# def soft_swap(array, soft_i_weights, soft_j_weights):
#     """
#     Perform a soft swap operation between two indices.

#     Args:
#         array (np.ndarray): The input array.
#         soft_i_weights (np.ndarray): Weights for the first index.
#         soft_j_weights (np.ndarray): Weights for the second index.

#     Returns:
#         np.ndarray: The updated array after soft swapping.
#     """
#     i_value = soft_get(array, soft_i_weights)
#     j_value = soft_get(array, soft_j_weights)

#     swapped_array = array.copy()
    
#     for k in range(len(array)):
#         swapped_array[k] = (
#            ? idk 
#         )
#     return swapped_array


def soft_set(array, soft_index_weights, value):
    """
    Perform a soft set operation on the array.

    Args:
        array (np.ndarray): The input array.
        soft_index_weights (np.ndarray): Probability distribution over the array indices.
        value (float): The value to softly set.

    Returns:
        np.ndarray: The updated array after the soft set.
    """
    updated_array = array.copy()
    for k in range(len(array)):
        updated_array[k] = array[k] * (1 - soft_index_weights[k]) + value * soft_index_weights[k]
    return updated_array


def soft_bubble_sort(array, temperature=1.0):
    """
    Perform a differentiable (soft) bubble sort on the array.

    Args:
        array (np.ndarray): The array to be softly sorted.
        temperature (float): Controls the smoothness of the sorting.

    Returns:
        np.ndarray: The softly sorted array.
    """
    n = len(array)
    sorted_array = array.copy()

    for i in range(n):
        for j in range(n - i - 1):
            soft_i_weights = soft_index(n, j, temperature)
            soft_j_weights = soft_index(n, j + 1, temperature)

            vi = soft_get(sorted_array, soft_i_weights)
            vj = soft_get(sorted_array, soft_j_weights)

            if vi > vj: 
                sorted_array = soft_swap(sorted_array, soft_i_weights, soft_j_weights)

    return sorted_array


def main():
    # Test soft bubble sort
    soft_arr = np.array([64, 34, 25, 12, 22, 11, 90])
    temperature = 1.0
    sorted_soft_arr = soft_bubble_sort(soft_arr, temperature)
    print("Soft Bubble Sort - Softly sorted array:", sorted_soft_arr, "\n")


if __name__ == "__main__":
    main()