# TabDDPM Model

## Model Overview

**TabDDPM (Tabular Denoising Diffusion Probabilistic Model)** is a generative diffusion model for mixed-type tabular data introduced by Kotelnikov, Baranchuk, Rubachev, and Babenko at ICML 2023. It models continuous features with a Gaussian diffusion process and categorical features with a Multinomial diffusion process, using a shared multi-layer perceptron (`MLPDiffusion`) denoiser conditioned on the target label.

This integration lives under `katabatic/experimental/models/tabddpm/` and is registered in `ModelRegistry` as `"tabddpm"`.

> **⚠️ Important — Local Fallback Warning (`_TABDDPM_EXTERNAL = False`):**
> `katabatic/experimental/models/tabddpm/models.py` attempts to import `GaussianMultinomialDiffusion` and `MLPDiffusion` from the external `tabddpm` package (`tabddpm.model.gaussian_multinomial_diffusion` and `tabddpm.model.modules`).
> When the external `tabddpm` package is **not** installed (`_TABDDPM_EXTERNAL = False`), Katabatic silently falls back to the lightweight stand-in classes in `katabatic/experimental/models/tabddpm/utils.py`.
>
> **Critical limitations of the local fallback (`utils.py`):**
> 1. **Zero categorical feature-learning gradients:** `utils.py::GaussianMultinomialDiffusion.mixed_loss()` computes categorical loss as `F.l1_loss(torch.round(x_cat_pred), torch.round(x_cat))`. Because `torch.round()` has zero derivative almost everywhere, categorical predictions receive zero gradient during backpropagation.
> 2. **Single-step denoiser sampling & hard clamping:** Instead of running a `num_timesteps`-step reverse Markov diffusion schedule, `utils.py::sample_all()` passes base Gaussian/uniform noise through `_denoise_fn` once, clamps numerical outputs to `[-3.0, 3.0]`, and rounds categorical indices (`torch.round(...).clamp_(min=0)`).
> 3. **Benchmark scripts emit a `RuntimeWarning` when `_TABDDPM_EXTERNAL` is `False`:** For paper-faithful benchmark results on categorical or mixed datasets (`car`, `nursery`, `adult`), install the external `tabddpm` backend; treat local-fallback runs as structural smoke tests only.

---

## Reference

Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023).
*TabDDPM: Modelling Tabular Data with Diffusion Models.*
Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202:17564–17579.
https://arxiv.org/abs/2209.15421

---

## Approach & Architecture

1. **Feature Separation & Encoding (`Tabddpm.train`)**:
   - Continuous columns are converted to `float32`. In the benchmark scripts (`benchmarks/examples/tabddpm/run_tabddpm_*.py`), continuous features are scaled to `[-1, 1]` using `sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))` before training and inverse-transformed after sampling so they stay within the local sampler's `[-3.0, 3.0]` clamp range.
   - Categorical columns (`cat_cols`) are integer-encoded to `0 .. K_i - 1` per column via `sklearn.preprocessing.LabelEncoder`.
   - For classification tasks (`_infer_task_type(y) == True`), target labels are integer-encoded via `LabelEncoder` and their empirical class prior `_class_dist` is stored for class-conditioned sampling.
2. **Denoiser Network (`MLPDiffusion`)**:
   - Concatenates numerical features, categorical index features, and (when `is_y_cond=True`) one-hot encoded class labels (`num_classes`).
   - Passes the vector through hidden linear layers configured by `d_layers` with ReLU activations and optional dropout.
3. **Training Loop & Exponential Moving Average (`use_ema=True`)**:
   - Trains for a fixed number of step iterations (`steps`, not epochs) using AdamW (`lr`, `weight_decay`).
   - Maintains an exponential moving average (`rate=0.999`) copy of the denoiser parameters (`_ema_model`) and swaps the EMA weights into the diffusion module after training completes.
4. **Sampling (`Tabddpm.sample`)**:
   - Draws target class labels from the empirical training class distribution `_class_dist`, generates synthetic feature tensors in micro-batches (`batch_size`), decodes categorical indices back to their original string/category labels via `_cat_label_encoders`, and appends the target column as `"label"`.

---

## Configuration & Hyperparameters

`Tabddpm` accepts hyperparameters via the `config: dict[str, Any]` argument passed to `model.train(...)`.

| Parameter | Type | Class Default (`_defaults`) | Pipeline Mode Default | Benchmark Script Setting | Description |
| --- | --- | --- | --- | --- | --- |
| `steps` | `int` | `10_000` | `200` | `2_000` | Total number of optimizer training steps (iterations). |
| `num_timesteps` | `int` | `1_000` | `100` | `1_000` | Number of diffusion timesteps in the schedule. |
| `batch_size` | `int` | `256` | `32` | `256` | Mini-batch size for training and sampling. |
| `lr` | `float` | `1e-3` | `1e-3` | `1e-3` | AdamW learning rate. |
| `weight_decay` | `float` | `1e-5` | `1e-5` | `1e-5` | AdamW weight decay penalty. |
| `gaussian_loss_type` | `str` | `"mse"` | `"mse"` | `"mse"` | Loss type for numerical diffusion (`"mse"` or `"kl"`). |
| `scheduler` | `str` | `"cosine"` | `"cosine"` | `"cosine"` | Noise schedule (`"cosine"` or `"linear"`). |
| `model_type` | `str` | `"mlp"` | `"mlp"` | `"mlp"` | Denoiser backbone architecture (`"mlp"`). |
| `d_layers` | `tuple[int, ...]` | `(256, 256, 256, 256)` | `(64, 64)` | `(256, 256, 256)` | Hidden layer widths of `MLPDiffusion`. |
| `dropout` | `float` | `0.0` | `0.0` | `0.0` | Dropout probability between hidden MLP layers. |
| `use_ema` | `bool` | `True` | `False` | `True` | Maintain and swap in EMA weights (`rate=0.999`) for sampling. |
| `eval_batches` | `int` | `50` | `3` | `10` | Number of batches averaged by `evaluate()`. |
| `seed` | `int` | `42` | `42` | `42` (`SEED`) | Random seed for Python, NumPy, and PyTorch (`_set_seed`). |

---

## Hardware & Runtime Notes

- **Device Selection**: `Tabddpm.__init__()` automatically selects `cuda` when `torch.cuda.is_available()` is True and falls back to `cpu` otherwise.
- **Step-Based Budget (`benchmarks/protocol.py`)**: Under the Katabatic benchmark protocol, `tabddpm` follows the `specific` budget policy because it trains by step count (`steps=2000` in benchmark scripts) rather than dataset epochs.
- **Memory & Speed**: With `d_layers=(256, 256, 256)` and `steps=2000`, CPU training completes in under a minute on small/medium datasets when running the local fallback, while full multi-step diffusion (`_TABDDPM_EXTERNAL = True`, `num_timesteps=1000`) benefits significantly from a CUDA GPU during reverse-diffusion sampling.

---

## Quick Start

### 1. Array / DataFrame Mode (Recommended — Honours `cat_cols`)

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from katabatic.experimental.models.tabddpm.models import Tabddpm

train_df = pd.read_csv("train_full.csv")
target_col = "class"
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
continuous_cols = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]

features = train_df.drop(columns=[target_col]).copy()
scaler = MinMaxScaler(feature_range=(-1, 1))
features[continuous_cols] = scaler.fit_transform(features[continuous_cols].astype(float))

model = Tabddpm()
model.train(
    features,
    train_df[target_col].astype(str),
    cat_cols=categorical_cols,
    config=dict(
        steps=2000,
        num_timesteps=1000,
        batch_size=256,
        d_layers=(256, 256, 256),
        use_ema=True,
        eval_batches=10,
        seed=42,
    ),
)

synthetic_df = model.sample(len(train_df), as_dataframe=True)
if "label" in synthetic_df.columns and target_col not in synthetic_df.columns:
    synthetic_df = synthetic_df.rename(columns={"label": target_col})
synthetic_df = synthetic_df.loc[:, train_df.columns].copy()
synthetic_df[continuous_cols] = scaler.inverse_transform(
    synthetic_df[continuous_cols].astype(float).clip(-1.0, 1.0)
)
```

### 2. Directory / Pipeline Mode (`x_train.csv` & `y_train.csv`)

```python
from katabatic.experimental.models.tabddpm.models import Tabddpm

model = Tabddpm()
model.train(
    "splits/adult",
    synthetic_dir="synthetic/adult/tabddpm",
    config=dict(steps=200, num_timesteps=100, batch_size=32, seed=42),
)
```

### Expected Files (Pipeline Mode)
- **Inputs** (in `dataset_dir`): `x_train.csv` and `y_train.csv`
- **Outputs** (in `synthetic_dir`): `x_synth.csv` and `y_synth.csv`
