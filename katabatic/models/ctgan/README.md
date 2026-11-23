# CTGAN

**CTGAN (Conditional Tabular Generative Adversarial Network)** is a conditional GAN approach for synthesizing mixed-type tabular data introduced by Xu, Skoularidou, Cuesta-Infante, and Veeramachaneni at NeurIPS 2019.

Katabatic's `CTGANModel` (`katabatic/models/ctgan/models.py`) adapts the core conditional-vector and categorical Gumbel-Softmax ideas from the paper into a lightweight, self-contained implementation with both a PyTorch WGAN / WGAN-GP backend (`backend="torch"`) and a fast, dependency-light NumPy fallback backend (`backend="numpy"`). See [Implementation Notes](#implementation-notes) for details on how it differs from the reference paper.

## Reference

Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
*Modeling Tabular Data using Conditional GAN.*
Advances in Neural Information Processing Systems (NeurIPS 2019).

## Overview

1. **Schema Inference & Encoding (`katabatic/models/ctgan/utils.py`)**:
   - `infer_schema(df)` classifies columns into `"categorical"` (object/category dtypes, or integer columns with `<= 20` distinct values or `< 5%` relative cardinality) and `"continuous"`.
   - Continuous columns are normalized to a standard normal distribution via `sklearn.preprocessing.QuantileTransformer(output_distribution="normal")`.
   - Categorical columns are one-hot encoded into contiguous blocks (`cat_blocks`).
2. **Conditioning Vector Construction**:
   - `build_conditioning(df, schema)` builds the concatenated one-hot conditioning space across all categorical columns (`cond_blocks`) and computes empirical category probabilities per column.
   - During training and unconditional sampling, `sample_conditions()` picks a categorical column uniformly at random and samples a category according to its empirical distribution.
3. **Adversarial Training (`backend="torch"`)**:
   - A feedforward generator and discriminator (`LocalMLP`) are trained for `epochs` epochs with `n_critic` critic updates per generator step using either weight clipping (`clip_value`) or WGAN-GP gradient penalty (`use_gradient_penalty=True`, `lambda_gp`).
   - Categorical output blocks are passed through `F.gumbel_softmax(..., tau=gumbel_tau, hard=False)` while continuous outputs use `torch.tanh`.
4. **NumPy Fallback (`backend="numpy"`)**:
   - When `backend="numpy"` is configured (or if PyTorch is unavailable), `fit()` records the fitted schema and quantile transformers, and `sample()` draws categoricals from their empirical distributions and continuous features from Gaussian distributions in quantile-transformed space.

## Architecture

- **Generator (`LocalMLP`)**: Input dimension `noise_dim + cond_dim`, hidden layers configured by `generator_hidden` (default `(256, 256)`) with ReLU activations, linear output layer of width `enc_dim`, followed by per-column Gumbel-Softmax (for categorical blocks) and `tanh` (for continuous columns).
- **Discriminator (`LocalMLP`)**: Input dimension `enc_dim + cond_dim`, hidden layers configured by `discriminator_hidden` (default `(256, 256)`) with ReLU activations, scalar linear output.
- **Optimizer**: Adam optimizer (`lr=2e-4`, `betas=(0.5, 0.9)`) for both networks.

## Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `epochs` | int | `100` | Number of training epochs over the dataset. |
| `batch_size` | int | `512` | Mini-batch size used by the PyTorch `DataLoader` (`drop_last=True`). |
| `noise_dim` | int | `128` | Dimensionality of the Gaussian latent noise vector `z`. |
| `generator_hidden` | tuple[int, ...] | `(256, 256)` | Hidden layer widths of the generator MLP. |
| `discriminator_hidden` | tuple[int, ...] | `(256, 256)` | Hidden layer widths of the discriminator MLP. |
| `lr` | float | `2e-4` | Adam learning rate for both generator and discriminator. |
| `betas` | tuple[float, float] | `(0.5, 0.9)` | Adam momentum hyperparameters `(beta1, beta2)`. |
| `lambda_gp` | float | `10.0` | WGAN-GP gradient penalty coefficient (used when `use_gradient_penalty=True`). |
| `use_gradient_penalty` | bool | `False` | Use WGAN-GP gradient penalty when `True`; otherwise use weight clipping (`clip_value`). |
| `clip_value` | float | `0.01` | Discriminator parameter clamp bound when `use_gradient_penalty=False`. |
| `n_critic` | int | `5` | Number of discriminator updates per generator update. |
| `gumbel_tau` | float | `0.2` | Gumbel-Softmax temperature applied to categorical output logits. |
| `seed` | int | `42` | Random seed for reproducibility. |
| `device` | str, optional | `None` | PyTorch device (`"cpu"` or `"cuda"`); auto-detects CUDA when omitted. |
| `backend` | str | `"torch"` | Execution backend (`"torch"` for neural WGAN training, `"numpy"` for empirical/quantile fallback). |

## Quick Start

```python
import pandas as pd

from katabatic.models.ctgan import CTGANModel

X_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

model = CTGANModel(
    epochs=100,
    batch_size=512,
    noise_dim=128,
    generator_hidden=(256, 256),
    discriminator_hidden=(256, 256),
    use_gradient_penalty=True,
    seed=42,
)

model.fit(X_train, y_train)

# Unconditional sampling
synthetic_df = model.sample(n_samples=1000)

# Single-column conditional sampling
conditional_df = model.sample(n_samples=200, conditional={"workclass": "Private"})
```

## Katabatic Integration

`CTGANModel.train(data_dir, *, synthetic_dir=None, artifact_state_dir=None, **kwargs)` reads `train_full.csv` (or `x_train.csv` and `y_train.csv`) from `data_dir`, fits the model, generates a synthetic dataset of equal row count, and writes `x_synth.csv`, `y_synth.csv`, and `metadata.json` to `synthetic_dir` (defaulting to `synthetic/<dataset_name>/ctgan`).

- **Registry**: Registered in `ModelRegistry` under `"ctgan"` (`supported: True`) and installable via `pip install katabatic[ctgan]`.
- **Artifact Persistence**: Declares `ARTIFACT_STATE_FILES = ("ctgan_state.pt",)` and supports `CTGANModel.load_from_ref(store, ref)` via `LocalArtifactStore` / `FsspecArtifactStore` inside `TrainTestSplitPipeline`. Because `LocalMLP` is defined inside `_build_torch_networks`, `load_from_ref()` reconstructs the network architecture from `cfg` before loading `generator_state` and `discriminator_state`.
- **Evaluation**: `model.evaluate(real_df, target_col=..., test_data=...)` evaluates the synthetic data across Katabatic's six evaluation dimensions (Utility/TSTR, Fidelity, Privacy, Diversity, Consistency, and Stability).

## Implementation Notes

Where Katabatic's `CTGANModel` differs from the NeurIPS 2019 paper and `sdv-dev/CTGAN`:

- **Quantile normalization instead of VGM mode-specific normalization**: Continuous columns are transformed to a Gaussian distribution via `QuantileTransformer(output_distribution="normal")` rather than fitting a Variational Gaussian Mixture Model per continuous column.
- **Empirical condition sampling instead of log-frequency training-by-sampling**: Condition vectors during training and sampling are drawn from the empirical category frequencies of a uniformly chosen categorical column (`sample_conditions`), without a cross-entropy conditional generator loss term or PacGAN input packing.
- **DataLoader drop-last behavior**: The PyTorch `DataLoader` uses `drop_last=True` with `batch_size=512` by default; for small datasets (`< 512` rows), pass a smaller `batch_size` (e.g., `batch_size=32` or `64`) so batches are not dropped during epoch loops.

## Benchmark Protocol & Runtime Notes

Under the Katabatic benchmark protocol (`benchmarks/protocol.py` / `benchmarks/examples/ctgan/`):
- **Policy**: `standard` — `epochs = epochs_for(dataset)` (`100` epochs for small/medium datasets: `car`, `nursery`, `magic`; `150` epochs for large datasets: `adult`, `shuttle`).
- **Seed & Split**: `seed = 42` (`SEED`), 80/20 stratified train/test split (`TEST_SIZE = 0.2`), `batch_size = 512`.
- **Measured Runtimes & Benchmark Scores**: Recorded via `benchmarks/runtime.py` and `benchmarks/run_all.py` across `car`, `adult`, `magic`, `nursery`, and `shuttle` (table populated from consolidated `benchmarks/results/` leaderboard output).

