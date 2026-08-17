<div align="center">

<img src="assets/goodfire_logo.png" height="64" alt="Goodfire AI"/>

# Block-Sparse Featurizers

**Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds**


</div>

---

<img src="assets/figure_1.png" width="100%" alt="Block-sparse featurizers capture the internal geometry of concepts."/>

> **Figure 1.** Current featurizers use *directions* as primitives — the SAE on the right renders the rabbit's ears and face as two such directions. The BSF lifts the primitive from directions to *regions of activation space*, recovering the ear and face concepts as manifolds whose internal coordinate (hue heatmap) reports where within the concept a given activation lies. The face concept resolves into fine-grained facial features (left/right of nose and eyes), while the ear feature varies smoothly along the rabbit's ear.

---

## What is a Block-Sparse Featurizer?

Standard sparse autoencoders (SAEs) decompose neural representations into sparse sums of **directions** — one atom per concept. But recent work shows that concepts in vision models are often realized as **low-dimensional subspaces and manifolds**, not single directions. A SAE applied to such a manifold is forced to tile it with many separate atoms, fragmenting what should be a single coherent concept.

**Block-Sparse Featurizers (BSFs)** attempt to adress this by lifting the unit of sparsity from individual directions to **blocks** — small groups of directions spanning a subspace. This is motivated by a generative model in which each activation is a sparse sum of points drawn from low-dimensional manifolds (an *additive mixture of manifolds*). The natural Bayesian prior for this model is block sparsity: sparse *across* concepts, dense *within* each active concept.

A BSF code carries two quantities per concept where a SAE carries only one:

| Quantity | Meaning | Visualization |
|---|---|---|
| `‖z_g‖₂` (block norm) | How strongly the concept is present | Single-color heatmap, like a SAE |
| `z_g` (block coordinate) | *Where* within the concept the activation lies | Hue map (first 3 PCA axes → RGB) |

The internal coordinate is precisely what SAEs discard — it is what gives BSFs their ability to recover concept manifolds and their intrinsic geometry.

## Three Featurizer Variants

| Featurizer | Encoder | Sparsity mechanism | Decoder |
|---|---|---|---|
| **Vanilla BSF** | Free linear `(W, b)` | Block TopK (keep `l0` blocks of largest norm) | Free `D` |
| **Grassmannian BSF** | Tied: `z_g = γ · x Dg^T` | Block TopK | Orthonormal frames `Dg Dg^T = I` |
| **Group Lasso BSF** | Free linear `(W, b)` | Block JumpReLU gate `H(‖a_g‖ − θ)`, θ learned by STE; sparsity set as a target L0 | Free `D` |

All three subclass `bsf.BSF` and share a single interface (`encode` / `loss` / decoder), so one trainer and one visualizer work for all of them.

---

## Quickstart

```python
import bsf, numpy as np, torch, einops
from bsf import data

# Load 300 rabbit images and compute DINOv3 patch activations (GPU needed)
images = data.load_rabbit_images('rabbit.npz')
acts   = data.dino_activations(images)          # (300, 196, 768)

# Center and scale
acts = acts - data.POS_MEAN
x    = einops.rearrange(acts, 'n p d -> (n p) d')
x    = x / np.sqrt((x ** 2).sum(1).mean()) * np.sqrt(x.shape[1])
grid = data.patch_grid(acts.shape[1])           # 14

# Train a Grassmannian BSF with 3D blocks
model = bsf.GrassmannianBSF(d=768, n_groups=256, group_size=3, l0=16)
bsf.train(model, x, epochs=60)

# Encode and visualize top concepts
z     = model.encode(torch.as_tensor(x, dtype=torch.float32, device='cuda')).detach().cpu().numpy()
atoms = model.atoms().cpu().numpy()
heat  = np.linalg.norm(z, axis=-1)
fire, energy = (heat > 1e-6).sum(0), (heat ** 2).sum(0)
top   = [g for g in np.argsort(-energy) if fire[g] >= 100][:20]

bsf.viz.plot_concepts(z, atoms, images, top, grid, n_img=10)
```

---

## Notebooks

Three end-to-end starter notebooks, one per featurizer variant:

| Notebook | Featurizer |
|---|---|
| [`starters/01_grassmannian.ipynb`](starters/01_grassmannian.ipynb) | Grassmannian BSF |
| [`starters/02_group_lasso.ipynb`](starters/02_group_lasso.ipynb) | Group Lasso BSF |
| [`starters/03_vanilla.ipynb`](starters/03_vanilla.ipynb) | Vanilla BSF |

Each notebook locates the repo root automatically and runs end-to-end. Computing DINOv3 activations requires a GPU; training and plotting run on CPU.

---

## Authors

**Thomas Fel** · **Matthew Kowal** · **Mozes Jacobs** · **Dron Hazra** · **Usha Bhalla** *(equal contribution)*

Lee Sharkey · Lucius Bushnaq · Satchel Grant · Tal Haklay · Thomas Icard ·
Can Rager · Michael Pearce · Daniel Wurgaft · Aiden Swann · Fenil Doshi ·
Siddharth Boppana · Curt Tigges · Nick Cammarata · Thomas Serre · Vasudev Shyam · Owen Lewis

Thomas McGrath · **Jack Merullo** · **Ekdeep Singh Lubana** · **Atticus Geiger** *(equal senior contribution)*

<img src="assets/goodfire_logo.png" height="32" alt="Goodfire"/>

Goodfire &nbsp;·&nbsp; Harvard University &nbsp;·&nbsp; Stanford University &nbsp;·&nbsp; Brown University

---

## Citation

```bibtex
@article{fel2025bsf,
  title   = {Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds},
  author  = {Fel, Thomas and Kowal, Matthew and Jacobs, Mozes and Hazra, Dron and Bhalla, Usha and
             Sharkey, Lee and Bushnaq, Lucius and Grant, Satchel and Haklay, Tal and Icard, Thomas and
             Rager, Can and Pearce, Michael and Wurgaft, Daniel and Swann, Aiden and Doshi, Fenil and
             Boppana, Siddharth and Tigges, Curt and Cammarata, Nick and Serre, Thomas and Shyam, Vasudev and
             Lewis, Owen and McGrath, Thomas and Merullo, Jack and Lubana, Ekdeep Singh and Geiger, Atticus},
  journal = {arXiv preprint},
  year    = {2025},
  url     = {https://github.com/goodfire-ai/block-sparse-featurizer}
}
```
