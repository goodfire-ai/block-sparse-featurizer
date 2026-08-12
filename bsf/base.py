"""Shared base for the three block-sparse featurizers.

A *concept* is a block of `group_size` latent dims; a featurizer carves the
`d`-dim activation space into `n_groups` such blocks. All three featurizers
share the same linear decoder -- stack the gated codes and multiply by the
decoder atoms (one (group_size, d) block per concept) -- so the only things a
featurizer defines are how it `encode`s an input into gated per-block codes and
what `loss` it optimises.
"""
import torch
import torch.nn as nn
from einops import rearrange


def group_topk(group_norms, l0):
    """Per-sample block TopK (the projection Pi_l): in each row keep the `l0`
    blocks of largest norm and zero the rest. Returns a {0, 1} mask (N, G)."""
    idx = group_norms.topk(l0, dim=-1).indices
    return torch.zeros_like(group_norms).scatter_(1, idx, 1.0)


def unit_blocks(W, n_groups, group_size, eps=1e-8):
    """Normalise each decoder atom to unit L2 norm."""
    Wb = rearrange(W, '(g s) d -> g s d', g=n_groups, s=group_size)
    norm = Wb.norm(dim=2, keepdim=True).clamp_min(eps)
    Wb_normed = Wb / norm
    return rearrange(Wb_normed, 'g s d -> (g s) d')


class BSF(nn.Module):
    """Block-Sparse Featurizer.

    Subclasses implement `encode(x) -> (N, n_groups, group_size)` and
    `loss(x, target) -> (scalar, info)`. They expose their decoder either as a
    `W_dec` parameter (the default) or by overriding `decoder_atoms`.
    """

    def __init__(self, d, n_groups, group_size):
        super().__init__()
        self.d = d
        self.n_groups = n_groups
        self.group_size = group_size

    # provided by subclasses
    def encode(self, x):
        raise NotImplementedError

    def loss(self, x, target=None):
        raise NotImplementedError

    # (n_groups * group_size, d) decoder matrix
    def decoder_atoms(self):
        return self.W_dec

    def decode(self, z):
        return rearrange(z, 'n g s -> n (g s)') @ self.decoder_atoms()

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    # per-concept decoder atoms (n_groups, group_size, d) for the viz
    def atoms(self):
        W = self.decoder_atoms().detach()
        return rearrange(W, '(g s) d -> g s d', g=self.n_groups, s=self.group_size)

    # unit-L2 each decoder atom (no-op for an orthonormal decoder)
    @torch.no_grad()
    def normalize_decoder(self):
        if hasattr(self, 'W_dec'):
            self.W_dec.data.copy_(
                unit_blocks(self.W_dec.data, self.n_groups, self.group_size))
