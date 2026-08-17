"""Group-lasso BSF — signed JumpReLU and a per-block threshold.

A free linear encoder produces signed per-block codes (no ReLU). A block fires
when its L2 norm is above a per-block threshold theta; the full signed code is
kept (no magnitude shrinkage, which collapses under a norm-constrained decoder).
theta is learned by a straight-through estimator -- the block analogue of
JumpReLU: the gate H(||a_g|| - theta) is exact in the forward pass and a kernel
`K` evaluated at ||a_g|| - theta carries the gradient in the backward pass.

`K` is selectable (`bsf.kernels`) and gaussian by default.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import BSF, unit_blocks
from .kernels import gaussian_kernel, get_kernel


class BlockJumpReLU(torch.autograd.Function):
    """Hard block gate H(||a_g|| - theta) with a kernel pseudo-derivative.

    Backward sends -K to theta and +K to ||a_g||: the gate is forward-exact and
    both of its arguments receive the derivative of the kernel-smoothed gate.
    """

    @staticmethod
    def forward(ctx, gn, theta, bandwidth, kernel=None):
        ctx.save_for_backward(gn, theta)
        ctx.bw = float(bandwidth)
        ctx.kernel = kernel or gaussian_kernel
        return (gn > theta).to(gn.dtype)

    @staticmethod
    def backward(ctx, g):
        gn, theta = ctx.saved_tensors
        gk = g * ctx.kernel(gn - theta, ctx.bw)
        return gk, -gk.sum(0), None, None


class GroupLassoBSF(BSF):
    """Group-lasso block-sparse featurizer.

    Args:
        target_l0: requested mean number of active blocks per token. Enforced as
            a constraint by dual ascent, not by hand-tuning `coef`.
        bandwidth: JumpReLU kernel bandwidth, as a multiple of std(||a_g||).
        kernel: name (see `bsf.kernels.KERNELS`) or callable for the JumpReLU
            pseudo-derivative. Gaussian by default; the heavier-tailed cauchy
            leaves several times as many blocks dead at the same sparsity.
        coef: initial value of the Lagrange multiplier (the dual variable finds
            its own level from here; this is not a sparsity knob).
        dual_lr: step size for dual ascent on log(lambda).
    """

    def __init__(
        self,
        d,
        n_groups,
        group_size=3,
        coef=1e-2,
        target_l0=16,
        gain=10.0,
        bandwidth=0.25,
        kernel='gaussian',
        dual_lr=3e-2,
    ):
        super().__init__(d, n_groups, group_size)
        self.coef = coef
        self.target_l0 = target_l0
        self.gain = gain
        self.bw_mult = float(bandwidth)
        self.kernel_name = kernel if isinstance(kernel, str) else kernel.__name__
        self.kernel = get_kernel(kernel)
        self.dual_lr = dual_lr

        W = unit_blocks(torch.randn(n_groups * group_size, d), n_groups, group_size)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.t().clone())  # tied init
        self.b_enc = nn.Parameter(torch.zeros(n_groups * group_size))

        # per-block threshold, theta = softplus(gain * raw_theta) >= 0
        self.raw_theta = nn.Parameter(torch.zeros(n_groups))
        # bandwidth in absolute units, tracked from the block-norm scale
        self.register_buffer('bandwidth', torch.ones(()))
        self.register_buffer('inited', torch.zeros((), dtype=torch.bool))
        # dual variable for the L0 constraint (hand-updated, not Adam's)
        self.register_buffer('log_lambda', torch.tensor(math.log(coef)))

    def theta(self):
        return F.softplus(self.gain * self.raw_theta)

    @torch.no_grad()
    def init_theta(self, gn):
        """Cold-start: place theta so ~`target_l0` blocks fire on the first batch."""
        q = 1.0 - self.target_l0 / self.n_groups
        thr = torch.quantile(gn.flatten().float(), q).clamp_min(1e-3)
        self.raw_theta.copy_(torch.log(torch.expm1(thr)) / self.gain)
        self.bandwidth.copy_(gn.std().clamp_min(1e-6) * self.bw_mult)
        self.inited.fill_(True)

    def gate(self, gn):
        if self.training and not bool(self.inited):
            self.init_theta(gn)
        if self.training:
            with torch.no_grad():
                # EMA of the block-norm scale, so the bandwidth keeps meaning the
                # same thing as the encoder grows
                self.bandwidth.mul_(0.99).add_(
                    0.01 * gn.detach().std().clamp_min(1e-6) * self.bw_mult)
        return BlockJumpReLU.apply(gn, self.theta(), float(self.bandwidth),
                                   self.kernel)

    @torch.no_grad()
    def dual_step(self, l0):
        """Dual ascent on log(lambda) for the constraint E[L0] = target_l0.

        The relative error keeps the step scale-free; clamping it to +-1 stops the
        cold-start spike (L0 = n_groups) from blowing lambda up in a single step.
        """
        err = (l0 - self.target_l0) / max(self.target_l0, 1e-8)
        self.log_lambda.add_(self.dual_lr * max(-1.0, min(1.0, float(err))))
        self.log_lambda.clamp_(math.log(1e-8), math.log(1e2))

    def preact(self, x):
        a = x @ self.W_enc + self.b_enc
        return rearrange(a, 'n (g s) -> n g s', g=self.n_groups, s=self.group_size)

    def encode(self, x):
        a = self.preact(x)
        return a * self.gate(a.norm(dim=-1)).unsqueeze(-1)

    def loss(self, x, target=None):
        target = x if target is None else target
        a = self.preact(x)
        gate = self.gate(a.norm(dim=-1))
        z = a * gate.unsqueeze(-1)
        recon = (target - self.decode(z)).pow(2).mean()
        l0 = gate.sum(-1).mean()
        lam = float(self.log_lambda.exp())
        l0_val = float(l0.detach())
        if self.training:
            self.dual_step(l0_val)
        return recon + lam * l0, {'recon': float(recon.detach()), 'l0': l0_val,
                                  'lambda': lam}
