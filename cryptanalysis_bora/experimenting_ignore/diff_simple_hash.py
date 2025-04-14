from simple_hash import *
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

"""
Differentiable approximation of x % m using the fractional part method.
This version is compatible with autograd.
"""
def differentiable_mod(x, m):
    # Use fractional part: x - floor(x)
    return m * (x / m - np.floor(x / m))

def main():
    return
