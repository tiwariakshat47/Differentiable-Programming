import numpy as np
# import scipy as sp
from scipy.stats import norm


def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)


def softrank(l):
    # Implementation of Taylor Softranking (2008).
    variance = 2


    soft_ranks = []
    for j in range(len(l)):
        r = 0
        for i in range(len(l)):
            if i != j:
                # Equation (9) in Taylor (2008)
                # norm.cdf integrates from -inf to x
                # -inf => 0 is same as 0 => inf because Normal is symmetrical.
                pi_ij = norm.cdf(0, l[i] - l[j], 2 * variance)
                r += pi_ij
        soft_ranks.append(r)

    return np.array(soft_ranks)
        

# Big Q: is there some other way to use softrank other than MSE?
def softrank_mse_loss(pred, target):
    # print(softrank(pred))
    # print(softrank(target))
    return mse_loss(softrank(pred), softrank(target))
    


# def corrcoef_loss(x: torch.Tensor, y: torch.Tensor):
#     x = x - x.mean()
#     y = y - y.mean()
#     x = x / x.norm()
#     y = y / y.norm()
#     return -(x * y).sum()


if __name__ == '__main__':

    # Test softrank
    a = [3,2,1]
    print(f'softrank {a}: {softrank(a)})')
    a = [1,2,3]
    print(f'softrank {a}: {softrank(a)})')
    a = [10,2,3]
    print(f'softrank {a}: {softrank(a)})')

    # Test softrank loss
    a = [3,2,1]
    print(f'\nsoftrank_loss {a}: {softrank_mse_loss(a, sorted(a))}')
    a = [10,2,1]
    print(f'softrank_loss {a}: {softrank_mse_loss(a, sorted(a))}')
    a = [4,2,10]
    print(f'softrank_loss {a}: {softrank_mse_loss(a, sorted(a))}')




    # softrank_loss(arr, sorted(arr))
