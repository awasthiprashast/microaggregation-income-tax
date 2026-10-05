"""
Reproducible synthetic income & tax population.

The generator draws a population whose *joint* structure resembles real
taxpayer microdata, because micro-aggregation results only mean something if
the data has realistic correlations and tails:

* demographics: age, gender, marital status (age-dependent), dependents
  (depends on marital status and age), education, occupation (age- and
  education-dependent), region and city tier;
* gross income: log-normal with an occupation baseline, an education premium,
  a concave experience curve, a city-tier premium, plus a Pareto tail for the
  self-employed and professionals (very high incomes are the hardest records
  to protect);
* deductions: standard deduction, investment / insurance / home-loan items whose
  take-up rises with income;
* capital gains: zero for most filers, log-normal for the rest;
* taxable income, tax paid and tax bracket computed from the stylised schedule
  in ``microagg.tax`` (so the dataset satisfies real accounting identities).

Finally a small amount of *dirt* is injected - missing values, invalid entries
and duplicated rows - so that the validation/imputation code is exercised.

Every random draw comes from one ``numpy.random.Generator`` seeded by ``seed``,
so the same (n, seed) always yields the identical file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import tax
from .config import CATEGORY_LEVELS, DEFAULT_N_RECORDS, RANDOM_SEED


def _choice(rng: np.random.Generator, levels, probs, size=None):
    probs = np.asarray(probs, dtype=float)
    return rng.choice(np.asarray(levels, dtype=object), p=probs / probs.sum(), size=size)


def _generate_clean(n: int, rng: np.random.Generator) -> pd.DataFrame:
    # ---------------- demographics ---------------- #
    age = np.clip(np.round(rng.normal(41, 13, n)), 21, 85).astype(int)

    gender = _choice(rng, ["Male", "Female", "Other"], [0.62, 0.373, 0.007], n)
    region = _choice(
        rng, CATEGORY_LEVELS["region"], [0.22, 0.24, 0.16, 0.22, 0.10, 0.06], n
    )
    city_tier = _choice(rng, ["Tier-1", "Tier-2", "Tier-3"], [0.35, 0.40, 0.25], n)

    # Marital status: probability of being single falls with age, widowhood rises.
    p_single = 1.0 / (1.0 + np.exp((age - 29) / 3.5))
    p_widowed = np.clip((age - 55) / 120, 0, 0.25)
    p_divorced = np.full(n, 0.03)
    p_married = np.clip(1 - p_single - p_widowed - p_divorced, 0, 1)
    marital_probs = np.column_stack([p_divorced, p_married, p_single, p_widowed])
    marital_probs /= marital_probs.sum(axis=1, keepdims=True)
    u = rng.random(n)[:, None]
    marital_idx = (u > np.cumsum(marital_probs, axis=1)).sum(axis=1)
    marital_status = np.asarray(CATEGORY_LEVELS["marital_status"], dtype=object)[marital_idx]

    lam = np.where(marital_status == "Married", 1.5, np.where(marital_status == "Single", 0.15, 0.8))
    lam = lam + np.where((age > 45) & (age < 65), 0.4, 0.0)        # dependent parents
    num_dependents = np.clip(rng.poisson(lam), 0, 6)

    education = _choice(
        rng, CATEGORY_LEVELS["education"], [0.12, 0.18, 0.42, 0.24, 0.04], n
    )
    edu_rank = pd.Series(education).map(
        {lvl: i for i, lvl in enumerate(CATEGORY_LEVELS["education"])}
    ).to_numpy()

    # Occupation depends on age (retirement) and education (professions).
    occupation = np.empty(n, dtype=object)
    working_levels = [
        "Salaried-Private", "Salaried-Government", "Self-Employed", "Professional", "Freelancer",
    ]
    for i in range(n):
        if age[i] >= 60:
            occupation[i] = _choice(
                rng, ["Retired", "Self-Employed", "Professional", "Salaried-Private"],
                [0.65, 0.17, 0.08, 0.10],
            )
        else:
            prof_weight = 0.14 if edu_rank[i] >= 3 else (0.06 if edu_rank[i] == 2 else 0.0)
            occupation[i] = _choice(
                rng, working_levels, [0.45, 0.15, 0.20, prof_weight, 0.10]
            )

    # ---------------- income ---------------- #
    base = pd.Series(occupation).map({
        "Salaried-Private": np.log(650_000),
        "Salaried-Government": np.log(750_000),
        "Self-Employed": np.log(600_000),
        "Professional": np.log(1_200_000),
        "Freelancer": np.log(450_000),
        "Retired": np.log(380_000),
    }).to_numpy()
    sigma = pd.Series(occupation).map({
        "Salaried-Private": 0.45, "Salaried-Government": 0.30, "Self-Employed": 0.70,
        "Professional": 0.55, "Freelancer": 0.60, "Retired": 0.40,
    }).to_numpy()
    edu_effect = np.array([-0.35, -0.20, 0.0, 0.25, 0.40])[edu_rank]
    exp_years = np.clip(age - 22, 0, None)
    experience = np.where(occupation == "Retired", 0.0, 0.035 * exp_years - 0.0006 * exp_years**2)
    tier_effect = pd.Series(city_tier).map({"Tier-1": 0.25, "Tier-2": 0.0, "Tier-3": -0.20}).to_numpy()

    log_income = base + edu_effect + experience + tier_effect + rng.normal(0, 1, n) * sigma
    gross_income = np.exp(log_income)

    # Pareto tail: a few business owners / professionals earn far more.
    tail_eligible = np.isin(occupation, ["Self-Employed", "Professional"])
    in_tail = tail_eligible & (rng.random(n) < 0.04)
    gross_income[in_tail] *= 1.0 + rng.pareto(1.6, in_tail.sum()) * 3.0
    gross_income = np.round(np.maximum(gross_income, 120_000), -2)

    # ---------------- deductions ---------------- #
    salaried_like = np.isin(occupation, ["Salaried-Private", "Salaried-Government", "Retired"])
    standard = np.where(salaried_like, 50_000, 0)
    p_invest = 1.0 / (1.0 + np.exp(-(np.log(gross_income) - np.log(500_000)) * 2.0))
    investment = np.where(
        rng.random(n) < p_invest,
        np.minimum(150_000, rng.lognormal(np.log(0.08 * gross_income), 0.5)),
        0.0,
    )
    health = np.where(rng.random(n) < 0.55, rng.uniform(10_000, 50_000, n), 0.0)
    home_loan = np.where(
        (age >= 28) & (age <= 58) & (rng.random(n) < 0.25),
        np.minimum(200_000, rng.lognormal(np.log(120_000), 0.5)),
        0.0,
    )
    deductions = standard + investment + health + home_loan
    deductions = np.round(np.minimum(deductions, 0.5 * gross_income), -2)

    taxable_income = np.maximum(gross_income - deductions, 0.0)

    # ---------------- capital gains & tax ---------------- #
    p_gains = np.clip(0.10 + 0.12 * (np.log(gross_income) - np.log(300_000)), 0.05, 0.75)
    capital_gains = np.where(
        rng.random(n) < p_gains,
        np.round(rng.lognormal(np.log(0.12 * gross_income), 1.0), -2),
        0.0,
    )
    tax_paid = np.round(tax.compute_tax(taxable_income, capital_gains))

    ids = rng.permutation(np.arange(1_000_000, 1_000_000 + 10 * n, 10))[:n]
    return pd.DataFrame({
        "taxpayer_id": [f"TP{x:08d}" for x in ids],
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "num_dependents": num_dependents,
        "education": education,
        "occupation": occupation,
        "region": region,
        "city_tier": city_tier,
        "gross_income": gross_income,
        "deductions": deductions,
        "taxable_income": taxable_income,
        "capital_gains": capital_gains,
        "tax_paid": tax_paid,
        "tax_bracket": tax.tax_bracket(taxable_income),
    })


def _inject_dirt(df: pd.DataFrame, rng: np.random.Generator, missing_rate: float,
                 invalid_rate: float, duplicate_rate: float) -> pd.DataFrame:
    """Add MCAR missing values, invalid entries and duplicate rows."""
    df = df.copy()
    n = len(df)
    # Object dtype so that a few non-numeric strings can be injected.
    for col in ("age", "num_dependents", "gross_income", "deductions", "capital_gains"):
        df[col] = df[col].astype(object)

    missing_cols = {
        "age": missing_rate, "num_dependents": missing_rate, "education": missing_rate,
        "marital_status": missing_rate, "city_tier": missing_rate,
        "deductions": missing_rate, "capital_gains": missing_rate,
        "gross_income": missing_rate / 3, "taxable_income": missing_rate / 3,
    }
    for col, rate in missing_cols.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    def pick(rate):
        return rng.random(n) < rate

    df.loc[pick(invalid_rate), "age"] = 150            # out-of-range typo
    df.loc[pick(invalid_rate / 2), "age"] = "?"        # non-numeric entry
    df.loc[pick(invalid_rate), "num_dependents"] = 25  # implausible value
    neg = pick(invalid_rate)
    df.loc[neg, "gross_income"] = -pd.to_numeric(df.loc[neg, "gross_income"], errors="coerce")
    df.loc[pick(invalid_rate / 2), "gender"] = " male "  # inconsistent formatting

    n_dup = max(1, int(round(duplicate_rate * n)))
    dups = df.iloc[rng.choice(n, n_dup, replace=False)]
    df = pd.concat([df, dups], ignore_index=True)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def generate_income_tax_data(
    n: int = DEFAULT_N_RECORDS,
    seed: int = RANDOM_SEED,
    missing_rate: float = 0.03,
    invalid_rate: float = 0.004,
    duplicate_rate: float = 0.002,
    dirty: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic income/tax dataset.

    Parameters
    ----------
    n : number of distinct taxpayers (duplicates are added on top when dirty).
    seed : seed of the random generator; identical (n, seed) => identical data.
    missing_rate, invalid_rate, duplicate_rate : amount of injected dirt.
    dirty : if False, return the clean population (useful for tests).
    """
    if n < 10:
        raise ValueError("n must be at least 10")
    rng = np.random.default_rng(seed)
    df = _generate_clean(n, rng)
    if dirty:
        df = _inject_dirt(df, rng, missing_rate, invalid_rate, duplicate_rate)
    return df
