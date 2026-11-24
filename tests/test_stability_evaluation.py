"""Unit tests for katabatic/evaluate/stability/evaluation.py."""

import pandas as pd
import pytest

from katabatic.evaluate.stability.evaluation import StabilityEvaluation


class _SeededModel:
    """Returns a small, seed-dependent perturbation of real_data."""

    def __init__(self, real_data):
        self._real = real_data

    def sample(self, n_samples, seed=0):
        df = self._real.copy().iloc[:n_samples].reset_index(drop=True)
        df["num"] = df["num"] + seed
        return df


class _UnseededModel:
    """Does not accept a seed kwarg at all -> triggers the TypeError fallback."""

    def __init__(self, real_data):
        self._real = real_data

    def sample(self, n_samples):
        return self._real.copy().iloc[:n_samples].reset_index(drop=True)


class _FailingModel:
    def sample(self, n_samples, seed=0):
        raise RuntimeError("boom")


class _NoneReturningModel:
    def sample(self, n_samples, seed=0):
        return None


class _EmptyFrameModel:
    def sample(self, n_samples, seed=0):
        return pd.DataFrame(columns=["cat", "num"])


class _PartiallyFailingModel:
    """Fails on seed=1, succeeds on other seeds."""

    def __init__(self, real_data):
        self._real = real_data

    def sample(self, n_samples, seed=0):
        if seed == 1:
            raise RuntimeError("seed 1 failed")
        df = self._real.copy().iloc[:n_samples].reset_index(drop=True)
        df["num"] = df["num"] + seed
        return df


class _NonDeterministicSeedModel:
    """Accepts seed kwarg but ignores it and mutates counter each call."""

    def __init__(self, real_data):
        self._real = real_data
        self._calls = 0

    def sample(self, n_samples, seed=0):
        self._calls += 1
        df = self._real.copy().iloc[:n_samples].reset_index(drop=True)
        df["num"] = df["num"] + self._calls * 5
        return df


class _WrongTypeModel:
    def sample(self, n_samples, seed=0):
        return [1, 2, 3]


class _UnseededAndFailingModel:
    """No seed support, and the unseeded fallback call also raises."""

    def sample(self, n_samples):
        raise RuntimeError("still broken")


@pytest.fixture
def real_data():
    return pd.DataFrame({"cat": ["a", "b"] * 10, "num": list(range(20))})


def test_seeded_model_produces_varying_scores(real_data):
    ev = StabilityEvaluation(
        real_data,
        _SeededModel(real_data),
        n_runs=3,
        seeds=[0, 1, 2],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    results = ev.evaluate()

    assert results["n_runs"] == 3
    assert results["seeds"] == [0, 1, 2]
    assert len(results["per_run_scores"]) == 3
    assert 0.0 <= results["stability_score"] <= 1.0
    assert "fidelity" in results["mean_scores"]
    assert "diversity" in results["mean_scores"]
    assert results["seed_supported"] is True
    assert results["failed_runs"] == []
    assert results["warnings"] == []


def test_empty_real_data_raises():
    empty_df = pd.DataFrame(columns=["cat", "num"])
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        StabilityEvaluation(empty_df, model=object())


def test_invalid_n_runs_or_empty_seeds_raises(real_data):
    with pytest.raises(ValueError, match="n_runs must be at least 1"):
        StabilityEvaluation(real_data, model=object(), n_runs=0)

    with pytest.raises(ValueError, match="seeds cannot be empty"):
        StabilityEvaluation(real_data, model=object(), n_runs=1, seeds=[])


def test_seed_len_mismatch_raises():
    real = pd.DataFrame({"num": [1, 2, 3]})
    with pytest.raises(ValueError, match="len\\(seeds\\)"):
        StabilityEvaluation(real, model=object(), n_runs=3, seeds=[0, 1])


def test_default_seeds_and_n_samples(real_data):
    ev = StabilityEvaluation(real_data, _SeededModel(real_data), n_runs=2)
    assert ev.seeds == [0, 1]
    assert ev.n_samples == len(real_data)


def test_single_seed_case_records_warning(real_data):
    ev = StabilityEvaluation(
        real_data,
        _SeededModel(real_data),
        n_runs=1,
        seeds=[42],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    results = ev.evaluate()

    assert results["n_runs"] == 1
    assert results["seeds"] == [42]
    assert results["avg_std"] == 0.0
    assert results["stability_score"] == 1.0
    assert any("Only 1 sampling run completed" in w for w in results["warnings"])


def test_same_seed_gives_same_result(real_data):
    ev = StabilityEvaluation(
        real_data,
        _SeededModel(real_data),
        n_runs=2,
        seeds=[7, 7],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    results = ev.evaluate()

    assert results["same_seed_reproducible"] is True
    assert results["per_run_scores"][0] == results["per_run_scores"][1]
    assert results["avg_std"] == 0.0


def test_same_seed_inconsistency_is_flagged(real_data):
    ev = StabilityEvaluation(
        real_data,
        _NonDeterministicSeedModel(real_data),
        n_runs=2,
        seeds=[7, 7],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    results = ev.evaluate()

    assert results["same_seed_reproducible"] is False
    assert any("Seed 7 produced different synthetic data" in w for w in results["warnings"])


def test_unseeded_model_falls_back_and_makes_failure_visible(real_data, capsys):
    ev = StabilityEvaluation(
        real_data,
        _UnseededModel(real_data),
        n_runs=2,
        seeds=[0, 1],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    results = ev.evaluate()

    captured = capsys.readouterr()
    assert "does not accept a 'seed' argument" in captured.out
    assert results["seed_supported"] is False
    assert any("does not accept a 'seed' argument" in w for w in results["warnings"])
    assert results["stability_score"] == 1.0


def test_partial_run_failure_is_visible_in_results(real_data):
    ev = StabilityEvaluation(
        real_data,
        _PartiallyFailingModel(real_data),
        n_runs=3,
        seeds=[0, 1, 2],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    results = ev.evaluate()

    assert results["n_runs"] == 2
    assert results["seeds"] == [0, 2]
    assert len(results["failed_runs"]) == 1
    assert results["failed_runs"][0]["seed"] == 1
    assert "seed 1 failed" in results["failed_runs"][0]["error"]
    assert any("seed 1 failed" in w for w in results["warnings"])


def test_model_raising_is_skipped_and_failures_visible(real_data, capsys):
    ev = StabilityEvaluation(real_data, _FailingModel(), n_runs=2, seeds=[0, 1])
    results = ev.evaluate()

    assert results["stability_score"] == 0.0
    assert results["error"] == "All sampling runs failed."
    assert len(results["failed_runs"]) == 2
    assert results["failed_runs"][0]["error"] == "boom"
    captured = capsys.readouterr()
    assert "model.sample() raised" in captured.out


def test_model_returning_none_is_skipped_and_visible(real_data):
    ev = StabilityEvaluation(real_data, _NoneReturningModel(), n_runs=2, seeds=[0, 1])
    results = ev.evaluate()

    assert results["stability_score"] == 0.0
    assert results["error"] == "All sampling runs failed."
    assert len(results["failed_runs"]) == 2
    assert results["failed_runs"][0]["error"] == "sample() returned None"


def test_model_returning_empty_dataframe_is_skipped_and_visible(real_data):
    ev = StabilityEvaluation(real_data, _EmptyFrameModel(), n_runs=2, seeds=[0, 1])
    results = ev.evaluate()

    assert results["stability_score"] == 0.0
    assert results["error"] == "All sampling runs failed."
    assert len(results["failed_runs"]) == 2
    assert "empty DataFrame" in results["failed_runs"][0]["error"]


def test_unseeded_fallback_that_also_raises_is_skipped(real_data, capsys):
    ev = StabilityEvaluation(real_data, _UnseededAndFailingModel(), n_runs=1, seeds=[0])
    results = ev.evaluate()

    assert results["stability_score"] == 0.0
    assert results["error"] == "All sampling runs failed."
    assert results["seed_supported"] is False
    assert len(results["failed_runs"]) == 1
    assert results["failed_runs"][0]["error"] == "still broken"
    captured = capsys.readouterr()
    assert "does not accept a 'seed' argument" in captured.out
    assert "model.sample() raised: still broken" in captured.out


def test_model_returning_wrong_type_raises(real_data):
    ev = StabilityEvaluation(real_data, _WrongTypeModel(), n_runs=1, seeds=[0])
    with pytest.raises(TypeError, match="must return a pandas DataFrame"):
        ev.evaluate()


def test_sample_aligns_extra_columns_to_real_data(real_data):
    class _ExtraColsModel:
        def sample(self, n_samples, seed=0):
            df = real_data.copy()
            df["extra"] = "unused"
            return df

    ev = StabilityEvaluation(
        real_data,
        _ExtraColsModel(),
        n_runs=1,
        seeds=[0],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    synth = ev._sample(0)
    assert list(synth.columns) == ["cat", "num"]


def test_print_summary_runs_without_error(real_data, capsys):
    ev = StabilityEvaluation(
        real_data,
        _SeededModel(real_data),
        n_runs=2,
        seeds=[0, 1],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    ev.evaluate()
    captured = capsys.readouterr()
    assert "Stability Evaluation" in captured.out
    assert "Runs completed" in captured.out
