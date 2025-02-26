import numpy as np


def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def softrank_loss():
    pass



# def corrcoef_loss(x: torch.Tensor, y: torch.Tensor):
#     x = x - x.mean()
#     y = y - y.mean()
#     x = x / x.norm()
#     y = y / y.norm()
#     return -(x * y).sum()


if __name__ == '__main__':
    pass