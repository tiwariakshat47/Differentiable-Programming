import torch

def mse_loss(soft_sorted_array, classical_sorted_array):
    return torch.mean((soft_sorted_array - classical_sorted_array) ** 2)


def corrcoef_loss(x: torch.Tensor, y: torch.Tensor):
    x = x - x.mean()
    y = y - y.mean()
    x = x / x.norm()
    y = y / y.norm()
    return -(x * y).sum()


if __name__ == '__main__':
    pred = torch.tensor([3.,2.,1.])
    # pred = torch.tensor([1.,2.,3.4])
    target = torch.tensor([1.,2.,3.])
    print(spearman_loss(pred, target))