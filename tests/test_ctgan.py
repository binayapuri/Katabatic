"""Unit and regression tests for the CTGAN model and preprocessing utilities."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from katabatic.models.ctgan.models import CTGANModel
from katabatic.models.ctgan.utils import (
    build_conditioning,
    decode_batch,
    encode_df,
    fit_transformers,
    infer_categorical_columns,
    infer_schema,
    sample_conditions,
)


def _make_mixed_dataframe(n_rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.normal(38.0, 10.0, size=n_rows).round(2),
            "hours_per_week": rng.normal(40.0, 6.0, size=n_rows).round(2),
            "workclass": rng.choice(["Private", "Self-emp", "Gov"], size=n_rows),
            "education": rng.choice(["Bachelors", "Masters", "HS-grad"], size=n_rows),
            "income": rng.choice([0, 1], size=n_rows),
        }
    )


def test_ctgan_default_configuration():
    model = CTGANModel()

    assert model.cfg["epochs"] == 100
    assert model.cfg["batch_size"] == 512
    assert model.cfg["noise_dim"] == 128
    assert model.cfg["generator_hidden"] == [256, 256]
    assert model.cfg["discriminator_hidden"] == [256, 256]
    assert model.cfg["backend"] == "torch"
    assert model.is_fitted is False
    assert CTGANModel.get_required_dependencies() == ["torch", "sklearn"]


def test_infer_schema_and_encode_decode_roundtrip():
    df = _make_mixed_dataframe(40)
    cat_cols = infer_categorical_columns(df)

    assert "workclass" in cat_cols
    assert "education" in cat_cols
    assert "income" in cat_cols
    assert "age" not in cat_cols

    schema = infer_schema(df)
    fit_transformers(df, schema)
    enc, cat_blocks, output_order = encode_df(df, schema)

    assert output_order == list(df.columns)
    assert set(cat_blocks.keys()) == {"workclass", "education", "income"}
    assert enc.shape[0] == len(df)

    decoded = decode_batch(enc, schema)
    assert list(decoded.columns) == list(df.columns)
    assert len(decoded) == len(df)
    assert list(decoded["workclass"]) == list(df["workclass"])


def test_build_and_sample_conditioning_vectors():
    df = _make_mixed_dataframe(30)
    schema = infer_schema(df)
    cond, cond_blocks = build_conditioning(df, schema)

    assert cond.shape[0] == 30
    assert set(cond_blocks.keys()) == {"workclass", "education", "income"}

    sampled = sample_conditions(15, schema, cond_blocks)
    assert sampled.shape == (15, cond.shape[1])
    np.testing.assert_allclose(sampled.sum(axis=1), np.ones(15))


def test_sample_before_fit_raises_runtime_error():
    model = CTGANModel(backend="numpy")

    with pytest.raises(RuntimeError, match="Call train\\(\\) before sample\\(\\)"):
        model.sample(10)


def test_ctgan_numpy_backend_fit_and_sample_preserves_schema():
    df = _make_mixed_dataframe(50)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = CTGANModel(epochs=2, batch_size=16, backend="numpy", seed=42)
    model.fit(X, y)

    assert model.is_fitted is True
    synth_df = model.sample(n_samples=25)

    assert synth_df.shape == (25, 5)
    assert list(synth_df.columns) == list(df.columns)
    assert not synth_df.isnull().any().any()
    assert set(synth_df["workclass"].unique()).issubset({"Private", "Self-emp", "Gov"})


def test_ctgan_train_writes_synthetic_csvs_and_metadata(tmp_path):
    df = _make_mixed_dataframe(40)
    data_dir = tmp_path / "data"
    synth_dir = tmp_path / "synth"
    data_dir.mkdir()

    df.iloc[:, :-1].to_csv(data_dir / "x_train.csv", index=False)
    df.iloc[:, -1:].to_csv(data_dir / "y_train.csv", index=False)

    model = CTGANModel(epochs=1, batch_size=16, backend="numpy", seed=42)
    model.train(str(data_dir), synthetic_dir=str(synth_dir))

    x_synth_path = synth_dir / "x_synth.csv"
    y_synth_path = synth_dir / "y_synth.csv"
    meta_path = synth_dir / "metadata.json"

    assert x_synth_path.is_file()
    assert y_synth_path.is_file()
    assert meta_path.is_file()

    x_synth = pd.read_csv(x_synth_path)
    y_synth = pd.read_csv(y_synth_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert len(x_synth) == 40
    assert len(y_synth) == 40
    assert list(x_synth.columns) == ["age", "hours_per_week", "workclass", "education"]
    assert list(y_synth.columns) == ["income"]
    assert meta["schema"]["label"] == "income"
