"""One trainer for all three featurizers.

It is featurizer-agnostic: it only relies on `model.loss(x, target)` and
`model.normalize_decoder()`. Plain Adam, full passes over the activation matrix.

With `snr > 0` it trains as a denoiser: the input is corrupted with Gaussian
noise of std `snr * std(x)`, but the model still reconstructs the clean `x`.
"""
import torch


@torch.no_grad()
def recon_r2(model, x):
    """Fraction of variance explained by the (clean) reconstruction."""
    x_hat = model(x)[0]
    ss_res = (x - x_hat).pow(2).sum()
    ss_tot = (x - x.mean(0, keepdim=True)).pow(2).sum()
    return float(1.0 - ss_res / ss_tot.clamp_min(1e-12))


@torch.no_grad()
def l0_dead(model, x):
    """Mean active blocks per token, and number of blocks that never fire."""
    # (N, G)
    active = model.encode(x).norm(dim=-1) > 1e-6
    l0 = float(active.float().sum(1).mean())
    dead = int((~active.any(0)).sum())
    return l0, dead


def train(model, x, *, epochs=40, lr=4e-4, batch_size=2048, snr=0.1,
          warmup=0.05, device=None, log_every=5):
    """Train `model` on activations `x` (N, d). Returns the trained model.

    `warmup` fraction of epochs used for linear LR warm-up, followed by cosine
    decay to 0.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x = torch.as_tensor(x, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches_per_epoch = max(1, x.shape[0] // batch_size)
    total_steps = epochs * n_batches_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps,
        pct_start=warmup, anneal_strategy='cos', div_factor=25, final_div_factor=1e4)
    n = x.shape[0]

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        n_batches = 0
        for i in range(0, n - batch_size + 1, batch_size):
            xb = x[perm[i:i + batch_size]].to(device)
            xb_in = xb if snr <= 0 else xb + snr * xb.std() * torch.randn_like(xb)
            # reconstruct clean xb
            loss, _ = model.loss(xb_in, target=xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            model.normalize_decoder()
            running += float(loss.item())
            n_batches += 1

        if ep == 1 or ep == epochs or ep % log_every == 0:
            model.eval()
            sub = x[torch.randperm(n)[:20_000]].to(device)
            r2 = recon_r2(model, sub)
            l0, dead = l0_dead(model, sub)
            print(f'epoch {ep:3d}/{epochs}   loss={running / n_batches:.4f}   '
                  f'R2={r2:.4f}   L0={l0:.1f}   dead={dead}/{model.n_groups}', flush=True)
    return model
