# GReaT

**GReaT (Generation of Realistic Tabular Data with Pretrained Transformer Language Models)** is an autoregressive language-model approach for synthetic tabular data generation introduced by Borisov, Seßler, Leemann, Pawelczyk, and Kasneci at ICLR 2023.

Katabatic's `GReaT` (`katabatic/models/great/models.py`) converts tabular rows into natural-language text sequences (`"<col> is <val>, <col> is <val>, ..."`), fine-tunes a HuggingFace causal language model (such as `gpt2` or `distilgpt2`) with random feature-order permutation, and generates new rows by sampling text tokens and parsing them back into a structured DataFrame.

> **Hardware & runtime:** Fine-tuning and sampling a causal Transformer language model is computationally intensive. A GPU is strongly recommended for real datasets; for local CPU smoke tests, use a small checkpoint (`llm="distilgpt2"` or `"gpt2"`), few epochs (`epochs=1` to `3`), and small sample sizes.

## Reference

Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., & Kasneci, G. (2023).
*Language Models are Realistic Tabular Data Generators.*
International Conference on Learning Representations (ICLR 2023).

## Overview

1. **Textual Serialization & Feature-Order Permutation (`katabatic/models/great/utils.py`)**:
   - `GReaTDataset` converts each DataFrame row into a text string where each feature is formatted as `"<column_name> is <value>"`, separated by commas.
   - During data collation (`GReaTDataCollator`), the column order of each row is randomly permuted so the causal language model learns conditional distributions across arbitrary feature orderings.
2. **Causal LM Fine-Tuning**:
   - Initializes `AutoTokenizer` and `AutoModelForCausalLM` from the specified HuggingFace checkpoint (`llm`), setting `pad_token = eos_token`.
   - Optionally applies parameter-efficient LoRA fine-tuning (`efficient_finetuning="lora"`) via `peft` (`r=16`, `lora_alpha=32`, `target_modules=["c_attn"]`, `lora_dropout=0.05`).
   - Trains using `GReaTTrainer` (a subclass of HuggingFace `Trainer`) and `TrainingArguments`.
3. **Guided & Legacy Sampling**:
   - **Legacy batch sampling (`guided_sampling=False`, default)**: Builds starting prompts via `CategoricalStart`, `ContinuousStart`, or `RandomStart` (drawn from the empirical distribution of the starting column), generates token sequences in batches of size `k`, and filters/parses valid rows via `_convert_text_to_tabular_data` until `n_samples` complete rows are collected.
   - **Guided feature-by-feature sampling (`guided_sampling=True`)**: Generates each row column-by-column in either fixed or random feature order (`random_feature_order=True`), retrying or falling back to empirical training values if a generated value cannot be parsed cleanly.

## Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `llm` | str | *(required)* | HuggingFace pretrained causal LM checkpoint (e.g., `"distilgpt2"` or `"gpt2"`). |
| `experiment_dir` | str | `"trainer_great"` | Output directory where HuggingFace `Trainer` checkpoints are written. |
| `epochs` | int | `100` | Number of fine-tuning epochs (`num_train_epochs`). |
| `batch_size` | int | `8` | Per-device training batch size (`per_device_train_batch_size`). |
| `efficient_finetuning` | str | `""` | Set to `"lora"` to enable int8/LoRA parameter-efficient fine-tuning via `peft`. |
| `float_precision` | int, optional | `None` | Number of decimal places to round floating-point values to before text encoding (`None` keeps full precision). |
| `report_to` | list[str], optional | `None` | HuggingFace logging integrations (defaults to `[]`, disabling Weights & Biases prompts). |
| `**train_kwargs` | Any | `{}` | Additional keyword arguments forwarded directly to `transformers.TrainingArguments`. |

### Sampling Parameters (`model.sample(...)`)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_samples` | int, optional | `None` | Number of synthetic rows to generate (defaults to the training row count). |
| `start_col` | str, optional | `""` | Column used to seed generation prompts (defaults to the final/target column). |
| `start_col_dist` | dict \| list, optional | `None` | Custom distribution for the starting column; uses empirical training distribution when `None`. |
| `temperature` | float | `0.7` | Autoregressive sampling temperature (`> 0`). |
| `k` | int | `100` | Batch size of parallel text sequences generated per sampling step in legacy mode. |
| `max_length` | int | `100` | Maximum token length per generated sequence. |
| `drop_nan` | bool | `False` | Drop rows with missing/unparsed values during legacy text-to-tabular conversion. |
| `device` | str | `"cpu"` | Device used during generation (`"cpu"` or `"cuda"`). |
| `guided_sampling` | bool | `False` | Use column-by-column guided prompt generation instead of full-row batch sampling. |
| `random_feature_order` | bool | `True` | Permute column generation order per row when `guided_sampling=True`. |
| `seed` | int, optional | `None` | Optional runtime random seed for reproducible sampling. |

## Quick Start

```python
import pandas as pd

from katabatic.models.great import GReaT

X_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

model = GReaT(
    llm="distilgpt2",
    epochs=5,
    batch_size=8,
    float_precision=2,
    report_to=[],
)

model.fit(X_train, y_train)

synthetic_df = model.sample(
    n_samples=100,
    temperature=0.7,
    k=16,
    device="cpu",
    seed=42,
)
```

## Katabatic Integration

`GReaT.train(data_dir, *, synthetic_dir=None, artifact_state_dir=None, **kwargs)` reads `x_train.csv` and `y_train.csv` from `data_dir`, concatenates them into a single frame, calls `fit()`, generates `len(train_df)` synthetic rows on CPU (`device="cpu"`), and writes `x_synth.csv` and `y_synth.csv` to `synthetic_dir` (defaulting to `synthetic/<dataset_name>/great`).

- **Registry**: Registered in `ModelRegistry` under `"great"` (`supported: True`) and installable via `pip install katabatic[great]`.
- **Artifact Persistence**: Declares `ARTIFACT_STATE_FILES = ("config.json", "model.pt")`. `_save_artifact_state()` serializes the metadata and schema configuration to `config.json` and saves `model.state_dict()` to `model.pt`; `GReaT.load_from_ref(store, ref)` rehydrates the fitted instance without re-running training.
- **Evaluation**: `model.evaluate(real_df, target_col=..., test_data=...)` scores the trained model across Katabatic's six evaluation dimensions (Utility/TSTR, Fidelity, Privacy, Diversity, Consistency, and Stability).

## Benchmark Protocol & Runtime Notes

Under the Katabatic benchmark protocol (`benchmarks/protocol.py` / `benchmarks/examples/great/`):
- **Policy**: `specific` — `llm = "gpt2"`, `seed = 42` (`SEED`, applied to sampling).
- **Dataset Budgets & Timing Investigation (Deliverable #13)**: `car`, `magic`, `nursery`, and `shuttle` currently run with `epochs = 2`, whereas `run_great_adult.py` sets `epochs = 310` (`batch_size = 128`). Because fine-tuning GPT-2 and rejection-sampling text clauses can exceed 1 hour on `adult`, separate timing of training, generation, and sampling is in progress to confirm the final recommended `adult` epoch budget and measured runtime table.

