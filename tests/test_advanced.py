"""Tests for the real-data loader and the contributions beyond plain MDAV."""

import numpy as np
import pandas as pd
import pytest

from microagg import advanced as A
from microagg import realdata as R


@pytest.fixture(scope="module")
def small():
    df = pd.read_csv(R.SAMPLE_CSV).head(600)
    return R.add_income_band(df)


def test_sample_is_complete_and_real_schema():
    df = pd.read_csv(R.SAMPLE_CSV)
    assert len(df) == 5000 and not df.isna().any().any()
    assert set(R.NUMERIC_QI + R.CATEGORICAL_QI + R.CONFIDENTIAL) <= set(df.columns)
    assert (df["total_income"] > 0).all() and df["age"].between(18, 90).all()


def test_reference_sample_is_disjoint_in_size_and_distribution():
    a, b = pd.read_csv(R.SAMPLE_CSV), pd.read_csv(R.REFERENCE_CSV)
    assert len(a) == len(b) == 5000
    assert abs(a["total_income"].median() / b["total_income"].median() - 1) < 0.15


def test_income_band_is_balanced(small):
    assert small["income_band"].value_counts().min() >= len(small) // 5 - 1


def test_emd_zero_for_global_distribution_and_positive_otherwise():
    g = np.full(5, 0.2)
    assert A.emd_ordered(np.ones(5), g) == pytest.approx(0.0)
    assert A.emd_ordered(np.array([5.0, 0, 0, 0, 0]), g) > 0.4


@pytest.mark.parametrize("method", ["mdav_all", "mdav_qi", "da_mdav", "random"])
def test_methods_are_k_anonymous(small, method):
    rel = A.anonymize(small, method, k=5)
    assert np.bincount(rel.labels).min() >= 5
    grouped = rel.anonymized.groupby(rel.labels).nunique()
    assert (grouped.max() <= 1).all()                 # every attribute constant within a group


@pytest.mark.parametrize("method", ["mdav_all", "mdav_qi", "da_mdav", "random"])
def test_means_preserved_exactly(small, method):
    rel = A.anonymize(small, method, k=5)
    for c in R.CONFIDENTIAL:
        assert rel.anonymized[c].mean() == pytest.approx(small[c].mean(), rel=1e-9)


def test_da_mdav_enforces_diversity_and_keeps_k(small):
    rel = A.anonymize(small, "da_mdav", k=5, l=3, t=0.3)
    stats = A.diversity_stats(small, rel.labels)
    assert stats["min_group"] >= 5
    assert stats["min_distinct_bands"] >= 3 and stats["max_emd"] <= 0.3 + 1e-9


def test_da_mdav_is_more_diverse_than_qi_mdav(small):
    plain = A.diversity_stats(small, A.anonymize(small, "mdav_qi", 5).labels)
    diverse = A.diversity_stats(small, A.anonymize(small, "da_mdav", 5).labels)
    assert diverse["mean_emd"] < plain["mean_emd"]


def test_standard_mdav_leaks_more_income_than_qi_space(small):
    full = A.fidelity(small, A.anonymize(small, "mdav_all", 5))["income_within10_%"]
    qi = A.fidelity(small, A.anonymize(small, "mdav_qi", 5))["income_within10_%"]
    assert full > qi


def test_centroid_noise_nonnegative_and_changes_values(small):
    base = A.anonymize(small, "mdav_qi", 5)
    noisy = A.anonymize(small, "mdav_qi", 5, noise_eps=1.0)
    assert (noisy.anonymized[list(R.CONFIDENTIAL)] >= 0).all().all()
    assert not np.allclose(base.anonymized["wages"], noisy.anonymized["wages"])
    assert np.bincount(noisy.labels).min() >= 5


def test_noise_is_reproducible(small):
    a = A.anonymize(small, "mdav_qi", 5, noise_eps=1.0, seed=3).anonymized
    b = A.anonymize(small, "mdav_qi", 5, noise_eps=1.0, seed=3).anonymized
    pd.testing.assert_frame_equal(a, b)


def test_ml_attack_baseline_beats_random_guess(small):
    out = A.ml_attack(small, None, small.sample(frac=1, random_state=1))
    assert out["ml_band_acc_%"] > 20.0
