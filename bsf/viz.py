"""Plotting the learned concepts.

`plot_concepts` draws one band per concept: on the left its firing cloud as a 3D
manifold (PCA of the contributions m_i, coloured by PCA coords -> RGB, point
size = norm, halo + shadowed square floor), and on the right a grid of the
images it fires hardest on -- each patch tinted by its PCA coordinate (hue =
where on the manifold it lies) with alpha = its norm.

The codes `z` (N, G, K) and decoder atoms (G, K, d) are passed in directly:
compute them in the notebook with `model.encode(x)` and `model.atoms()`.
"""
import numpy as np
import einops
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# registers the 3d projection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def pca_fit(c, k=3):
    """Mean and top-`k` principal axes of the contribution cloud `c` (n, d)."""
    mean = c.mean(0)
    comps = np.linalg.svd(c - mean, full_matrices=False)[2][:k]
    return mean, comps


def radial_clip(proj, pct=98.0):
    """Clip each point's norm to the `pct`-th percentile (floor 0): points beyond
    the cap are pulled radially onto it. Applied to BOTH the plotted geometry and
    the colour, so a few high-norm outliers neither stretch the cloud nor wash
    the colours. No low-end clip -- small norms are kept as-is."""
    norm = np.linalg.norm(proj, axis=1)
    cap = float(np.percentile(norm, pct))
    s = np.minimum(1.0, cap / np.maximum(norm, 1e-12))
    return proj * s[:, None]


def make_colorize(proj, per_axis=False, saturation=1.0):
    """Hue from the radial DIRECTION of each point (its unit vector p/‖p‖ mapped
    0.5 + 0.5·unit), so colour reflects *where on the manifold* a point lies and
    is independent of magnitude -- a small-norm point still gets a vivid hue.
    Magnitude (norm) is shown separately as intensity/alpha by the renderer.

    This avoids the gray-wash you get from dividing by a single max norm (where
    typical points, having norm << max, all collapse toward gray). `per_axis` is
    accepted for signature compatibility but unused -- direction colour is the
    same in both modes. `saturation` pushes colours away from gray (1.0 = none)."""
    def f(p):
        n = np.linalg.norm(p, axis=-1, keepdims=True)
        unit = p / np.maximum(n, 1e-8)
        return (0.5 + 0.5 * unit * saturation).clip(0, 1)
    return f


def manifold_ax(ax, proj, rgb, point_size=4.0, elev=16, azim=-60):
    """3D scatter: square floor below the cloud + shadow, no axes, halo behind
    each point. Every point is the same `point_size`; its INTENSITY (alpha) is
    its norm / max norm -- near-centre (low-norm) points fade, rim points are
    solid -- while `rgb` carries the direction hue. Each point gets a 2x larger
    translucent twin behind it as a halo."""
    # uniform size for every point
    s = point_size
    # intensity (alpha) = norm / max norm
    norm = np.linalg.norm(proj, axis=1)
    inten = (norm / max(norm.max(), 1e-8)).clip(0, 1)
    rgba = np.concatenate([np.asarray(rgb), inten[:, None]], axis=1)
    # floor clearly below the lowest point (margin scaled to the cloud size)
    span = np.ptp(proj, axis=0).max()
    floor = proj[:, 2].min() - 0.3 * span

    # square floor plane centred on the cloud
    cx, cy = proj[:, 0].mean(), proj[:, 1].mean()
    r = max(np.ptp(proj[:, 0]), np.ptp(proj[:, 1])) / 2 * 1.05 + 1e-8
    xx, yy = np.meshgrid([cx - r, cx + r], [cy - r, cy + r])
    ax.plot_surface(xx, yy, np.full_like(xx, floor), color='0.93', alpha=0.6,
                    shade=False, zorder=0)
    # shadow on the floor (alpha also scaled by intensity)
    ax.scatter(proj[:, 0], proj[:, 1], np.full(len(proj), floor),
               c=np.concatenate([np.full((len(proj), 3), 0.35), 0.05 * inten[:, None]], 1),
               s=s, edgecolors='none', depthshade=False)
    # halo: a 3x larger twin of each point at 0.1x its intensity
    halo = rgba.copy(); halo[:, 3] *= 0.1
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=halo, s=s * 3.0,
               edgecolors='none', depthshade=False)
    # the points (alpha = intensity)
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=rgba, s=s,
               edgecolors='none', depthshade=False)
    try:
        ax.axis('equal')
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def overlay(ax, image, z_patch, atoms_g, mean, comps, colorize, grid):
    """Tint each patch by the PCA->RGB of its contribution; alpha = its norm."""
    # (P, d)
    c = z_patch @ atoms_g
    # (P, 3)
    proj = (c - mean) @ comps.T
    rgb = colorize(proj)
    norm = np.linalg.norm(c, axis=1)
    alpha = (norm / max(norm.max(), 1e-8)).clip(0, 1)
    rgba = np.concatenate([rgb, alpha[:, None]], 1).reshape(grid, grid, 4)
    h, w = image.shape[:2]
    ax.imshow(image)
    ax.imshow(rgba, extent=(0, w, h, 0), interpolation='bicubic')
    ax.set_xticks([]); ax.set_yticks([])


_DEFAULT_VIEWS = [(16, -60), (16, 60), (16, 180)]


def plot_concepts(z, atoms, images, concepts, grid, n_img=10, ncol_img=5,
                  per_axis_rgb=False, clip=98.0, saturation=1.0,
                  drop_low_norm=0.0, max_points=5000,
                  point_size=4.0, concept_gap=0.6,
                  views=_DEFAULT_VIEWS):
    """One band per concept: 3D manifold views (left) + an `n_img` grid of overlays.

    z       (N, G, K)  gated codes (model.encode(x))
    atoms   (G, K, d)  decoder atoms (model.atoms())
    images  (n_imgs, H, W, 3)
    concepts           iterable of group indices to show
    drop_low_norm      fraction of lowest-norm firing points to discard before
                       fitting/plotting (cleans the gray center).
    point_size         uniform size of every 3D point (halo is 2x this).
    concept_gap        height (in row-units) of the blank spacer row inserted
                       between consecutive concept bands.
    views              list of (elev, azim) tuples — one subplot per view.
    """
    P = grid * grid
    # (N, G)
    heat = np.linalg.norm(z, axis=-1)
    zr = einops.rearrange(z, '(n p) g k -> n p g k', p=P)
    heat_img = einops.rearrange(heat, '(n p) g -> n p g', p=P)

    # image rows per concept
    nrow_img = int(np.ceil(n_img / ncol_img))
    # one column per manifold view
    mcol = len(views)
    ncols = mcol + ncol_img
    n_con = len(concepts)
    # one thin blank spacer row between consecutive concepts (none after last)
    rows_per = nrow_img + 1
    total_rows = rows_per * n_con - 1
    height_ratios = []
    for r in range(n_con):
        height_ratios += [1.0] * nrow_img
        if r < n_con - 1:
            height_ratios += [concept_gap]
    col_widths = [5.5] * mcol + [1.5] * ncol_img
    gs = GridSpec(total_rows, ncols, height_ratios=height_ratios,
                  width_ratios=col_widths, hspace=0.02, wspace=0.02)
    fig = plt.figure(figsize=(sum(col_widths), 3.5 * sum(height_ratios)))

    for r, g in enumerate(concepts):
        r0 = r * rows_per
        idx = np.where(heat[:, g] > 1e-6)[0]
        # drop the lowest-norm firing points
        if drop_low_norm > 0 and idx.size > 8:
            keep = heat[idx, g] >= np.quantile(heat[idx, g], drop_low_norm)
            idx = idx[keep]
        if idx.size < 8:
            for v in range(mcol):
                fig.add_subplot(gs[r0:r0 + nrow_img, v], projection='3d').set_axis_off()
            continue
        sub = idx if idx.size <= max_points else np.random.choice(idx, max_points, replace=False)
        # (n, d) contributions, then PCA-project to 3D
        c = z[sub, g, :] @ atoms[g]
        mean, comps = pca_fit(c, 3)
        proj = (c - mean) @ comps.T
        proj = radial_clip(proj, clip)
        colorize = make_colorize(proj, per_axis_rgb, saturation)
        rgb = colorize(proj)
        for v, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(gs[r0:r0 + nrow_img, v], projection='3d')
            manifold_ax(ax, proj, rgb, point_size=point_size, elev=elev, azim=azim)

        top_imgs = np.argsort(-(heat_img[:, :, g] ** 2).sum(1))[:n_img]
        for j, ii in enumerate(top_imgs):
            ax2 = fig.add_subplot(gs[r0 + j // ncol_img, mcol + j % ncol_img])
            overlay(ax2, images[ii], zr[ii, :, g, :], atoms[g], mean, comps, colorize, grid)

    fig.tight_layout()
    return fig
