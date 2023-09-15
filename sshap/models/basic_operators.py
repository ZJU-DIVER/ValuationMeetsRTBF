# modified from https://github.com/yzhao062/pytod/blob/main/pytod/models/basic_operators.py

import torch

# disable autograd since no grad is needed
torch.set_grad_enabled(False)


def top_k(A, k, dim=1, device='cpu'):
    """Returns the k the largest elements of the given input tensor along a given dimension.
    """
    if len(A.shape) == 1:
        dim = 0
    tk = torch.topk(A.to(device), k, dim=dim)
    return tk[0].cpu(), tk[1].cpu()


def bottom_k(A, k, dim=1, device='cpu'):
    if len(A.shape) == 1:
        dim = 0
    tk = torch.topk(A.to(device), k, dim=dim, largest=False)
    return tk[0].cpu(), tk[1].cpu()


def bottom_k_cpu(A, k, dim=1):
    if len(A.shape) == 1:
        dim = 0
    tk = torch.topk(A, k, dim=dim, largest=False)
    return tk[0], tk[1]


def hist_t(A, bins, v_min, v_max, device='cpu'):
    # e.g. a (pred res) = [0, 1, 1, 2, 0, 0], bins = 3, min = 0, max = 2
    # hist = [3, 2, 1]
    hist = torch.histc(A.to(device), bins=bins, min=v_min, max=v_max)
    bin_edges = torch.linspace(v_min, v_max, steps=bins + 1).to(device)
    return hist, bin_edges


def argmax_t(A, device='cpu'):
    v = torch.argmax(A.to(device))
    return v.cpu()


def batch_histogram(data_tensor, num_classes=-1):
    """
    Computes histograms of integral values, even if in batches (as opposed to torch.histc and torch.histogram).
    Arguments:
        data_tensor: a D1 x ... x D_n torch.LongTensor
        num_classes (optional): the number of classes present in data.
                                If not provided, tensor.max() + 1 is used (an error is thrown if tensor is empty).
    Returns:
        A D1 x ... x D_{n-1} x num_classes 'result' torch.LongTensor,
        containing histograms of the last dimension D_n of tensor,
        that is, result[d_1,...,d_{n-1}, c] = number of times c appears in tensor[d_1,...,d_{n-1}].
    """
    return torch.nn.functional.one_hot(data_tensor, num_classes).sum(dim=-2)


def batch_agg(data_tensor, num_classes=-1):
    """
    return unweighted voting result
    """
    hist_res = batch_histogram(data_tensor, num_classes)
    return torch.argmax(hist_res, dim=1).cpu()
