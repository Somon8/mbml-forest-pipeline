# MBML 42186 — Course Curriculum Reference
*DTU course. Use this document to understand which models and methods are in scope when generating notebooks for the group project.*

---

## Project context

The group project fits Bayesian regression models on Swedish forest raster data (1000×1000 grid, 10×10m cells, N≈10,000). Outcome is tree growth. Features include spatial coordinates, soil type, elevation, moisture, and others. The agreed modelling approach is **Bayesian Poisson regression with SVI in Pyro**, with spatial coordinates included as plain features in the linear predictor. All notebooks should stay within the models and inference methods listed below.

---

## Models in scope

### Bayesian Linear Regression (BLR)
Gaussian likelihood, Gaussian prior on weights. Already fitted as baseline. Useful for continuous outcomes.

### Bayesian Poisson Regression
Poisson likelihood with log link: `y_n ~ Poisson(exp(ω^T x_n))`, prior `ω ~ N(0, λI)`. The primary model for the project — appropriate for non-negative count or rate outcomes. Maps directly onto W5 material and the NYC taxi notebook.

### Bayesian Logistic Regression
Bernoulli likelihood with sigmoid link. For binary outcomes. Multi-class version uses softmax + Categorical. Not the primary project model but covered in W6.

### Hierarchical Models
Group-level parameters drawn from a shared hyperprior. Can be layered on top of any regression (linear, Poisson, logistic). Useful when data has natural groupings (e.g. spatial regions, soil zones). Covered in W6.

### Gaussian Mixture Models (GMM)
Unsupervised. Latent cluster assignments `z_n ~ Cat(π)`, observations `x_n ~ N(μ_{z_n}, Σ_{z_n})`. Covered in W4.

### Markov Models / HMMs
For sequential/temporal data. Markov assumption: current state depends only on previous state. Not relevant for this project. Covered in W7.

### Latent Dirichlet Allocation (LDA)
Topic model for document-word data. Not relevant for this project. Covered in W8.

### Probabilistic PCA (PPCA)
Linear dimensionality reduction with a probabilistic generative model. Latent variable `z_n ~ N(0,I)`, observation `x_n ~ N(Wz_n, σ²I)`. Covered in W11.

### Variational Autoencoder (VAE)
Non-linear extension of PPCA using neural networks. Encoder maps data → latent distribution; decoder maps latent → data. Covered in W11.

### Gaussian Processes (GP)
Non-parametric model placing a prior over functions. Captures smooth spatial/temporal structure via kernel. O(N³) — infeasible at N=10k without sparse approximations. **Not used directly in the project** — spatial coordinates are included as features in Poisson regression instead. Covered in W12.

---

## Inference methods in scope

### SVI (Stochastic Variational Inference)
Primary inference method for the project. Optimises the ELBO. Scales well to large N via minibatches. Use `AutoDiagonalNormal` or `AutoMultivariateNormal` as guide. Covered in W10.

### MCMC / NUTS
Gold-standard sampling-based inference. Pyro uses the No-U-Turn Sampler (NUTS). Gives correlated samples from the true posterior. Too slow for N=10k in practice — use SVI instead. Covered in W9.

### Exact inference (analytical posteriors)
Only possible for conjugate models (e.g. Gaussian likelihood + Gaussian prior). Not available for Poisson regression.

---

## Inference method selection guide

| Situation | Use |
|---|---|
| Large N, Poisson/logistic likelihood | SVI |
| Small N, want posterior samples | NUTS |
| Conjugate model | Analytical (exact) |
| Checking convergence | Compare SVI ELBO curve + NUTS traces |

---

## What NOT to generate

- Full GP models (O(N³), out of scope)
- Sparse GP / inducing point methods (beyond course scope)
- Neural network feature extractors
- Temporal/HMM models (not relevant to project)
- LDA / topic models (not relevant to project)
- VAEs (not relevant to project)
