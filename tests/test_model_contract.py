"""Model contract tests (B2):
1. Every model writes x_synth.csv and y_synth.csv with matching columns and row counts.
2. The generated synthetic label depends on the features (fails on the old ARF
   independent-label sampling code, passes on the fixed joint-label ARF and joint models).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from katabatic.models.arf.models import ARFModel
from katabatic.models.arf.utils import AdversarialRandomForest


def _training_table(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic classification dataset with strong feature-label dependence."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, size=n).astype(float)
    hours = rng.normal(40.0, 10.0, size=n).round(2)
    odds = (age - 40.0) / 8.0 + (hours - 40.0) / 10.0
    probs = 1.0 / (1.0 + np.exp(-odds))
    label = np.where(rng.random(n) < probs, "high", "low")
    return pd.DataFrame({"age": age, "hours": hours, "class": label})


def _write_training_dir(data_dir: Path, df: pd.DataFrame) -> None:
    """Write both train_full.csv and x_train.csv / y_train.csv for pipeline compatibility."""
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "train_full.csv", index=False)
    df.drop(columns=["class"]).to_csv(data_dir / "x_train.csv", index=False)
    df[["class"]].to_csv(data_dir / "y_train.csv", index=False)


def assert_label_depends_on_features(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    feature_col: str = "age",
    target_col: str = "class",
    min_ratio: float = 0.08,
    model_name: str = "model",
) -> None:
    """Assert that synthetic target labels preserve class-conditional feature dependence."""
    assert synthetic_df[target_col].nunique() >= 2, (
        f"{model_name} collapsed '{target_col}' to a single class"
    )
    real_gap = float(real_df.groupby(target_col)[feature_col].mean().diff().abs().iloc[-1])
    synth_gap = float(
        synthetic_df.groupby(target_col)[feature_col].astype(float).mean().diff().abs().iloc[-1]
    )
    assert synth_gap > min_ratio * real_gap, (
        f"{model_name} failed label-feature dependence check: "
        f"synth_gap={synth_gap:.4f} <= {min_ratio} * real_gap={real_gap:.4f}"
    )


def test_contract_fails_on_old_arf_independent_label_code():
    """Verify the contract check FAILS on the old ARF code that sampled y independently of X."""
    df = _training_table(n=1200, seed=0)
    X = df.drop(columns=["class"])
    y = df["class"]

    # Reproduce the old pre-fix ARF behaviour: fit on X only, then resample y independently.
    arf_engine = AdversarialRandomForest(
        num_trees=10,
        max_iters=3,
        random_state=42,
        verbose=False,
    )
    arf_engine.fit(X)
    arf_engine.forde()
    x_synth = arf_engine.forge(n=len(df))
    y_synth = y.sample(n=len(df), replace=True, random_state=42).reset_index(drop=True)

    old_arf_synthetic = x_synth.copy()
    old_arf_synthetic["class"] = y_synth.to_numpy()

    with pytest.raises(AssertionError, match="failed label-feature dependence check"):
        assert_label_depends_on_features(
            df,
            old_arf_synthetic,
            model_name="old_unfixed_arf",
        )


def test_contract_passes_on_fixed_arf_joint_label_code():
    """Verify the contract check PASSES on the fixed ARF logic that fits features + label jointly."""
    df = _training_table(n=1200, seed=0)

    # Fixed ARF behaviour: fit AdversarialRandomForest on the joint (features + label) table.
    arf_engine = AdversarialRandomForest(
        num_trees=10,
        max_iters=3,
        random_state=42,
        verbose=False,
    )
    arf_engine.fit(df)
    arf_engine.forde()
    fixed_arf_synthetic = arf_engine.forge(n=len(df))

    assert fixed_arf_synthetic.columns.tolist() == df.columns.tolist()
    assert_label_depends_on_features(df, fixed_arf_synthetic, model_name="fixed_joint_arf")


# (model_key, module_path, class_name, init_kwargs, checks_label_dependence)
CONTRACT_MODELS = [
    ("arf", "katabatic.models.arf.models", "ARFModel", {"verbose": False, "seed": 42}, False),
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
    df = _training_table(n=1000, seed=0)
    data_dir = tmp_path / f"dep_data_{model_key}"
    synth_dir = tmp_path / f"dep_synth_{model_key}"
    _write_training_dir(data_dir, df)

    model = _instantiate_model(module_path, class_name, init_kwargs)
    model.train(str(data_dir), synthetic_dir=str(synth_dir))

    x_synth = pd.read_csv(synth_dir / "x_synth.csv")
    y_synth = pd.read_csv(synth_dir / "y_synth.csv").iloc[:, 0]
    synthetic = x_synth.copy()
    synthetic["class"] = y_synth.values

    assert_label_depends_on_features(df, synthetic, model_name=model_key)
