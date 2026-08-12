"""Vanilla BSF.

A free linear encoder produces signed per-block codes; the block projection
Pi_l keeps the `l0` blocks of largest norm and zeroes the rest. Encoder and
decoder are untied; sparsity is enforced by construction (no penalty). With
`group_size=1` this is an ordinary absolute-TopK SAE.
"""
import torch
import torch.nn as nn
from einops import rearrange

from .base import BSF, group_topk, unit_blocks


class VanillaBSF(BSF):
    def __init__(self, d, n_groups, group_size=3, l0=16):
        super().__init__(d, n_groups, group_size)
        self.l0 = l0
        W = unit_blocks(torch.randn(n_groups * group_size, d), n_groups, group_size)
        self.W_dec = nn.Parameter(W)
        # tied init
        self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_groups * group_size))

    def encode(self, x):
        a = x @ self.W_enc + self.b_enc
        a = rearrange(a, 'n (g s) -> n g s', g=self.n_groups, s=self.group_size)
        # per-sample block TopK
        mask = group_topk(a.norm(dim=-1), self.l0)
        return a * mask.unsqueeze(-1)

    def loss(self, x, target=None):
        # target != x -> denoising
        target = x if target is None else target
        x_hat, _ = self(x)
        recon = (target - x_hat).pow(2).mean()
        return recon, {'recon': recon.item()}
