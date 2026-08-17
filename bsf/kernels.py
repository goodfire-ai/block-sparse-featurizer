"""Kernels for the block-JumpReLU pseudo-derivative.
"""
import torch


def rectangle_kernel(x, bandwidth):
    """Uniform on |x| <= h, zero outside. Mass 1."""
    h = bandwidth / 2
    return ((x >= -h) & (x <= h)).to(x.dtype) / bandwidth


def gaussian_kernel(x, bandwidth):
    """exp(-u^2 / 2) / (2h * sqrt(2 pi)),  u = x / h. Mass 0.5."""
    h = bandwidth / 2
    return torch.exp(-0.5 * (x / h) ** 2) / (bandwidth * (2.0 * torch.pi) ** 0.5)


def triangular_kernel(x, bandwidth):
    """(1 - |u|) on |u| <= 1, zero outside,  u = x / h. Mass 1."""
    h = bandwidth / 2
    u = x.abs() / h
    return (1 - u) * (u <= 1).to(x.dtype) / h


def cosine_kernel(x, bandwidth):
    """(1 + cos(pi x / bandwidth)) / 2 on |x| <= bandwidth. Mass `bandwidth`."""
    inside = (x.abs() <= bandwidth).to(x.dtype)
    return (1 + torch.cos(x * torch.pi / bandwidth)) / 2 * inside


def epanechnikov_kernel(x, bandwidth):
    """(3/4)(1 - u^2) on |u| <= 1, zero outside,  u = x / h. Mass 1."""
    h = bandwidth / 2
    u = x / h
    return (3 / 4) * (1 - u ** 2) * (u.abs() <= 1).to(x.dtype) / h


def quartic_kernel(x, bandwidth):
    """(15/16)(1 - u^2)^2 on |u| <= 1, zero outside,  u = x / h. Mass 1."""
    h = bandwidth / 2
    u = x / h
    return (15 / 16) * (1 - u ** 2) ** 2 * (u.abs() <= 1).to(x.dtype) / h


def silverman_kernel(x, bandwidth):
    """exp(-|u|/sqrt 2) sin(|u|/sqrt 2 + pi/4) / h,  u = x / h. Mass 2."""
    h = bandwidth / 2
    u = x.abs() / h
    return torch.exp(-u / 2 ** 0.5) * torch.sin(u / 2 ** 0.5 + torch.pi / 4) / h


def cauchy_kernel(x, bandwidth):
    """1 / (1 + u^2) / (pi h),  u = x / h. Mass 1, but heavy-tailed."""
    h = bandwidth / 2
    return 1 / (1 + (x / h) ** 2) / (torch.pi * h)


KERNELS = {
    'rectangle': rectangle_kernel,
    'gaussian': gaussian_kernel,
    'triangular': triangular_kernel,
    'cosine': cosine_kernel,
    'epanechnikov': epanechnikov_kernel,
    'quartic': quartic_kernel,
    'silverman': silverman_kernel,
    'cauchy': cauchy_kernel,
}


def get_kernel(kernel):
    """Look a kernel up by name; a callable passes straight through."""
    if callable(kernel):
        return kernel
    if kernel not in KERNELS:
        raise KeyError(f'unknown kernel {kernel!r}; choose from {sorted(KERNELS)}')
    return KERNELS[kernel]
