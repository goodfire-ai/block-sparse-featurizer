"""Grassmannian BSF.

Each concept is an orthonormal `group_size`-frame D_g (a point on a
Grassmannian). The constraint moves to the dictionary: encoder and decoder are
tied to the same frame, the code is z_g = gamma_g * x D_g^T (one learned scalar
gamma per concept compensates the energy lost by tying), and the block
projection Pi_l keeps
the `l0` frames of largest energy. A QR each forward keeps the frames
orthonormal, so the decoder needs no normalisation.
"""
import torch
import torch.nn as nn
from einops import rearrange

from .base import BSF, group_topk


class GrassmannianBSF(BSF):
    def __init__(self, d, n_groups, group_size=3, l0=16):
        super().__init__(d, n_groups, group_size)
        self.l0 = l0
        B = torch.randn(n_groups, d, group_size)
        # orthonormal columns
        B, _ = torch.linalg.qr(B)
        self.B_raw = nn.Parameter(B)
        # one gamma per concept, initialised to 1
        self.gamma = nn.Parameter(torch.ones(n_groups))

    # (n_groups * group_size, d), orthonormal within each concept
    def decoder_atoms(self):
        # (n_groups, d, group_size)
        B, _ = torch.linalg.qr(self.B_raw)
        return rearrange(B, 'g d s -> (g s) d')

    def encode(self, x):
        # (n_groups * group_size, d)
        atoms = self.decoder_atoms()
        # tied encoder = gamma_g * D, one gamma per concept
        z = rearrange(x @ atoms.t(), 'n (g s) -> n g s', g=self.n_groups, s=self.group_size)
        z = z * self.gamma[None, :, None]
        mask = group_topk(z.norm(dim=-1), self.l0)
        return z * mask.unsqueeze(-1)

    def loss(self, x, target=None):
        # target != x -> denoising
        target = x if target is None else target
        x_hat, _ = self(x)
        recon = (target - x_hat).pow(2).mean()
        return recon, {'recon': recon.item()}
