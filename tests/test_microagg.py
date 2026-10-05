"""Unit tests: python -m pytest -q"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microagg import privacy, release, safe_display, tax, utility
from microagg.anonymizer import anonymize
from microagg.config import AnonymizationConfig
from microagg.data_generator import generate_income_tax_data
from microagg.mdav import _k_nearest, aggregate, mdav, random_partition, univariate_groups
from microagg.preprocessing import DataValidationError, validate_and_clean

CFG = AnonymizationConfig()


@pytest.fixture(scope="module")
def clean():
    raw = generate_income_tax_data(n=600, seed=7)
    df, _ = validate_and_clean(raw, CFG.numeric_attributes, CFG.categorical_attributes)
    return df


# ----------------------------------------------------------------------------- MDAV
@pytest.mark.parametrize("n,k", [(10, 3), (11, 3), (12, 3), (50, 5), (101, 10), (7, 7), (13, 7)])
def test_mdav_group_sizes(n, k):
    X = np.random.default_rng(0).normal(size=(n, 3))
    labels = mdav(X, k)
    sizes = np.bincount(labels)
    assert labels.min() == 0 and (labels >= 0).all()          # every record assigned
    assert sizes.sum() == n
    assert sizes.min() >= k and sizes.max() <= 2 * k - 1      # MDAV size invariant
    assert (sizes > k).sum() <= 1                             # only the last group can be larger


def test_mdav_is_deterministic():
    X = np.random.default_rng(1).normal(size=(200, 4))
    assert np.array_equal(mdav(X, 5), mdav(X, 5))


def test_mdav_separates_obvious_clusters():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 0.1, size=(3, 2))
    b = rng.normal(100, 0.1, size=(3, 2))
    labels = mdav(np.vstack([a, b]), 3)
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_mdav_handles_duplicates():
    X = np.zeros((9, 2))                                      # all records identical
    sizes = np.bincount(mdav(X, 3))
    assert (sizes == 3).all()


def test_mdav_rejects_bad_input():
    with pytest.raises(ValueError):
        mdav(np.zeros((2, 2)), 3)
    with pytest.raises(ValueError):
        mdav(np.array([[np.nan, 1.0], [1, 2], [3, 4]]), 2)


def test_k_nearest_includes_anchor_and_breaks_ties_by_index():
    d = np.array([0.0, 0.0, 0.0, 0.0, 5.0])
    sel = _k_nearest(d, 2, anchor=3)
    assert 3 in sel and 0 in sel and len(sel) == 2


def test_random_and_univariate_group_sizes():
    for labels in (random_partition(23, 5, seed=0), univariate_groups(pd.Series(np.arange(23.0)), 5)):
        sizes = np.bincount(labels)
        assert sizes.min() >= 5 and sizes.sum() == 23


# ---------------------------------------------------------------- aggregation
def test_aggregation_preserves_means_and_linear_identity(clean):
    res = anonymize(clean, AnonymizationConfig(k=5))
    for c in CFG.numeric_attributes:
        assert res.anonymized[c].mean() == pytest.approx(res.original[c].mean(), rel=1e-9)
    a = res.anonymized
    # taxable = gross - deductions holds when the max(.,0) floor never binds
    if (res.original["gross_income"] >= res.original["deductions"]).all():
        assert np.allclose(a["gross_income"] - a["deductions"], a["taxable_income"], atol=1e-6)


def test_mdav_release_is_k_anonymous(clean):
    for k in (3, 5, 10):
        res = anonymize(clean, AnonymizationConfig(k=k))
        assert privacy.k_anonymity_level(res.anonymized, CFG.released_attributes) >= k


def test_mode_aggregation_tie_break():
    df = pd.DataFrame({"c": ["b", "a", "b", "a"]})
    out = aggregate(df, np.array([0, 0, 1, 1]), [], ["c"])
    assert list(out["c"]) == ["a", "a", "a", "a"]             # alphabetical tie-break


# --------------------------------------------------------------------- release
def test_release_guard_blocks_individual_ranking(clean):
    res = anonymize(clean, AnonymizationConfig(k=5, method="individual_ranking"))
    with pytest.raises(release.ReleaseValidationError):
        release.build_release(res)


def test_release_contains_no_identifiers_or_raw_values(clean):
    res = anonymize(clean, AnonymizationConfig(k=5))
    rel, checks = release.build_release(res)
    assert all(c.passed for c in checks if c.blocking)
    assert "taxpayer_id" not in rel.columns and "tax_bracket" not in rel.columns
    assert set(rel.columns) == {"release_id", *CFG.released_attributes}
    # row order is shuffled: released rows are not aligned with the source rows
    aligned = np.isclose(rel["gross_income"].to_numpy(), res.anonymized["gross_income"].round(2).to_numpy())
    assert aligned.mean() < 0.5
    # every released income value is shared by at least k records
    assert rel["gross_income"].value_counts().min() >= 5


# ---------------------------------------------------------------- preprocessing
def test_validation_handles_dirt():
    raw = generate_income_tax_data(n=400, seed=3, missing_rate=0.1, invalid_rate=0.02)
    df, rep = validate_and_clean(raw, CFG.numeric_attributes, CFG.categorical_attributes)
    assert not df[list(CFG.released_attributes)].isna().any().any()
    assert df["age"].between(18, 100).all()
    assert (df["gross_income"] >= 0).all()
    assert rep.duplicates_removed > 0
    assert set(df["gender"]) <= {"Male", "Female", "Other"}    # " male " normalised
    assert np.allclose(df["taxable_income"],
                       (df["gross_income"] - df["deductions"]).clip(lower=0))


def test_validation_rejects_missing_columns():
    with pytest.raises(DataValidationError):
        validate_and_clean(pd.DataFrame({"age": [30]}), ["age", "gross_income"], [])


def test_config_validation():
    with pytest.raises(ValueError):
        AnonymizationConfig(k=1).validate(100)
    with pytest.raises(ValueError):
        AnonymizationConfig(k=50).validate(10)


# ---------------------------------------------------------------- metrics
def test_information_loss_bounds(clean):
    il_small = utility.information_loss(anonymize(clean, AnonymizationConfig(k=3)))["il_percent"]
    il_large = utility.information_loss(anonymize(clean, AnonymizationConfig(k=20)))["il_percent"]
    assert 0 < il_small < il_large < 100


def test_linkage_risk_bounded_by_one_over_k(clean):
    res = anonymize(clean, AnonymizationConfig(k=5))
    d = privacy.distance_based_linkage(res, "all")
    assert d["max_record_risk"] <= 1 / 5 + 1e-12


def test_collusion_attack_recovers_exactly(clean):
    res = anonymize(clean, AnonymizationConfig(k=3))
    df = privacy.collusion_attack(res, fractions=(0.9,))
    row = df.iloc[0]
    assert row["exactly_recovered_%"] == pytest.approx(row["exposed_targets_%"])
    assert row["exposed_targets_%"] > 0


def test_tax_schedule():
    assert tax.compute_tax(np.array([700_000.0]), np.array([0.0]))[0] == 0      # rebate
    assert tax.compute_tax(np.array([1_000_000.0]), np.array([0.0]))[0] > 0
    assert list(tax.tax_bracket(np.array([100_000, 300_000, 300_001, 2_000_000]))) == ["0%", "0%", "5%", "30%"]


def test_safe_display_never_shows_exact_originals(clean):
    res = anonymize(clean, AnonymizationConfig(k=5))
    t = safe_display.comparison_table(res, n=20, seed=0)
    orig_cols = [c for c in t.columns if c.startswith("gross_income · original")]
    assert orig_cols and all(isinstance(v, str) and v.startswith("₹") for v in t[orig_cols[0]])
    summary = safe_display.safe_summary(clean, ["gross_income"])
    assert "min" not in summary.columns and "max" not in summary.columns
