"""Model contract tests: every model writes matching columns, and the label depends on the features."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _training_table(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic classification dataset with strong feature-label dependence."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, size=n).astype(float)
    hours = rng.normal(40.0, 10.0, size=n).round(2)
    odds = (age - 44.0) / 6.0 + (hours - 40.0) / 8.0
    probs = 1.0 / (1.0 + np.exp(-odds))
    label = np.where(rng.random(n) < probs, 1, 0)
    return pd.DataFrame({"age": age, "hours": hours, "class": label})


def _write_training_dir(data_dir: Path, df: pd.DataFrame) -> None:
    """Write both train_full.csv and x_train.csv / y_train.csv for pipeline compatibility."""
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "train_full.csv", index=False)
    df.drop(columns=["class"]).to_csv(data_dir / "x_train.csv", index=False)
    df[["class"]].to_csv(data_dir / "y_train.csv", index=False)


# (model_key, module_path, class_name, init_kwargs, checks_label_dependence)
# Note: HistogramModel and CTGANModel(backend="numpy") sample marginals independently
# by design, so they are checked for the column/output contract while joint models
# are checked for both the column contract and feature-label dependence.
CONTRACT_MODELS = [
    ("arf", "katabatic.models.arf.models", "ARFModel", {"verbose": False, "seed": 42}, True),
    ("ctgan_numpy", "katabatic.models.ctgan.models", "CTGANModel", {"backend": "numpy", "epochs": 1, "seed": 42}, False),
    ("gmm", "katabatic.experimental.models.gmm.models", "GMMModel", {"n_components": 2, "random_state": 42}, True),
    ("histogram", "katabatic.models.histogram.models", "HistogramModel", {"random_state": 42}, False),
    ("kde", "katabatic.models.kde.models", "KDESynthesizer", {"seed": 42}, True),
    ("naivebayes", "katabatic.models.naivebayes.models", "NaiveBayesModel", {"seed": 42}, True),
    ("privtree", "katabatic.models.privtree.models", "PrivTreeModel", {"epsilon": 5.0, "random_state": 42}, True),
    ("smote", "katabatic.models.smote.models", "SMOTEModel", {"random_state": 42}, True),
    ("tabebm", "katabatic.experimental.models.tabebm.models", "TabEBM", {"max_data_size": 300, "sgld_steps": 20, "seed": 42}, True),
    ("tabkde", "katabatic.models.tabkde.models", "TabKDEModel", {"seed": 42}, True),
]


def _instantiate_model(module_path: str, class_name: str, init_kwargs: dict):
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for {module_path}: {exc}")
    cls = getattr(mod, class_name)
    try:
        return cls(**init_kwargs)
    except ImportError as exc:
        pytest.skip(f"Optional runtime extra missing for {class_name}: {exc}")


@pytest.mark.parametrize(
    "model_key,module_path,class_name,init_kwargs,_checks_dep",
    CONTRACT_MODELS,
    ids=[m[0] for m in CONTRACT_MODELS],
)
def test_every_model_writes_matching_columns(
    tmp_path: Path,
    model_key: str,
    module_path: str,
    class_name: str,
    init_kwargs: dict,
    _checks_dep: bool,
):
    """Every model must write x_synth.csv and y_synth.csv with matching columns and row counts."""
    df = _training_table(n=300, seed=42)
    data_dir = tmp_path / f"data_{model_key}"
    synth_dir = tmp_path / f"synth_{model_key}"
    _write_training_dir(data_dir, df)

    model = _instantiate_model(module_path, class_name, init_kwargs)
    model.train(str(data_dir), synthetic_dir=str(synth_dir))

    x_synth_path = synth_dir / "x_synth.csv"
    y_synth_path = synth_dir / "y_synth.csv"

    assert x_synth_path.is_file(), f"{model_key} did not write x_synth.csv"
    assert y_synth_path.is_file(), f"{model_key} did not write y_synth.csv"

    x_synth = pd.read_csv(x_synth_path)
    y_synth = pd.read_csv(y_synth_path)

    assert x_synth.columns.tolist() == ["age", "hours"], (
        f"{model_key} x_synth columns {x_synth.columns.tolist()} != ['age', 'hours']"
    )
    assert y_synth.shape[1] == 1, f"{model_key} y_synth must have exactly 1 target column"
    assert len(x_synth) == len(df), f"{model_key} x_synth row count {len(x_synth)} != {len(df)}"
    assert len(y_synth) == len(df), f"{model_key} y_synth row count {len(y_synth)} != {len(df)}"
    assert not x_synth.isna().any().any(), f"{model_key} produced NaN values in x_synth"
    assert not y_synth.isna().any().any(), f"{model_key} produced NaN values in y_synth"

    sampled = model.sample(100)
    assert isinstance(sampled, pd.DataFrame), f"{model_key}.sample() must return a DataFrame"
    assert len(sampled) == 100
    assert sampled.columns.tolist()[:2] == ["age", "hours"]


@pytest.mark.parametrize(
    "model_key,module_path,class_name,init_kwargs",
    [(m[0], m[1], m[2], m[3]) for m in CONTRACT_MODELS if m[4]],
    ids=[m[0] for m in CONTRACT_MODELS if m[4]],
)
def test_synthetic_label_depends_on_features(
    tmp_path: Path,
    model_key: str,
    module_path: str,
    class_name: str,
    init_kwargs: dict,
):
    """Joint/conditional models must preserve feature-label dependence in synthetic output."""
    df = _training_table(n=800, seed=42)
    data_dir = tmp_path / f"dep_data_{model_key}"
    synth_dir = tmp_path / f"dep_synth_{model_key}"
    _write_training_dir(data_dir, df)

    model = _instantiate_model(module_path, class_name, init_kwargs)
    model.train(str(data_dir), synthetic_dir=str(synth_dir))

    x_synth = pd.read_csv(synth_dir / "x_synth.csv")
    y_synth = pd.read_csv(synth_dir / "y_synth.csv").iloc[:, 0]
    synthetic = x_synth.copy()
    synthetic["class"] = y_synth.values

    assert synthetic["class"].nunique() == 2, (
        f"{model_key} collapsed target to a single class"
    )

    real_gap = float(df.groupby("class")["age"].mean().diff().abs().iloc[-1])
    synth_gap = float(
        synthetic.groupby("class")["age"].astype(float).mean().diff().abs().iloc[-1]
    )

    # An independently sampled label produces a gap near zero; joint generators
    # preserve a meaningful fraction of the real class-conditional feature gap.
    assert synth_gap > 0.08 * real_gap, (
        f"{model_key} failed label-feature dependence check: "
        f"synth_gap={synth_gap:.4f} vs real_gap={real_gap:.4f}"
    )
