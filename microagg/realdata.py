"""
Real-data loader: US Census Bureau American Community Survey (ACS) 2022 1-year
Public Use Microdata Sample (PUMS), California person file.

Source (public, no login):
    https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/csv_pca.zip
Data dictionary: https://www.census.gov/programs-surveys/acs/microdata/documentation.html

The PUMS file already contains only de-identified, disclosure-limited person
records released by the Census Bureau. We draw a random sample of working-age
adults with positive personal income, recode a few variables into coarse
categories and adjust all dollar amounts to 2022 dollars with ADJINC.

Attribute roles
---------------
numeric quasi-identifiers  : age, hours_per_week
categorical quasi-identifiers : sex, marital_status, education, worker_class,
                             occupation_group, race, nativity, disability
confidential numeric       : total_income, wages, self_emp_income,
                             investment_income, retirement_income
sensitive class (derived)  : income_band (quintile of total_income) - used by the
                             l-diversity / t-closeness constraints and by the
                             attribute-disclosure attacks; never released.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "pums_ca" / "psam_p06.csv"
SAMPLE_CSV = ROOT / "data" / "acs_ca_2022_sample.csv"
REFERENCE_CSV = ROOT / "data" / "acs_ca_2022_reference.csv"

NUMERIC_QI = ("age", "hours_per_week")
CATEGORICAL_QI = ("sex", "marital_status", "education", "worker_class",
                  "occupation_group", "race", "nativity", "disability")
CONFIDENTIAL = ("total_income", "wages", "self_emp_income",
                "investment_income", "retirement_income")
MONETARY = CONFIDENTIAL
N_BANDS = 5

_USECOLS = ["AGEP", "WKHP", "SEX", "MAR", "SCHL", "COW", "SOCP", "RAC1P", "NATIVITY",
            "DIS", "PINCP", "WAGP", "SEMP", "INTP", "RETP", "ADJINC"]


def _education(s: pd.Series) -> pd.Series:
    bins = [-1, 15, 17, 20, 21, 24]
    labels = ["Below high school", "High school", "Some college / associate",
              "Bachelor", "Postgraduate"]
    return pd.cut(s, bins=bins, labels=labels).astype(str)


def _recode(raw: pd.DataFrame) -> pd.DataFrame:
    adj = raw["ADJINC"] / 1_000_000.0
    out = pd.DataFrame({
        "age": raw["AGEP"].astype(float),
        "hours_per_week": raw["WKHP"].fillna(0).astype(float),
        "sex": raw["SEX"].map({1: "Male", 2: "Female"}),
        "marital_status": raw["MAR"].map({1: "Married", 2: "Widowed", 3: "Divorced",
                                          4: "Divorced", 5: "Never married"}),
        "education": _education(raw["SCHL"]),
        "worker_class": raw["COW"].map({1: "Private", 2: "Private", 3: "Government",
                                        4: "Government", 5: "Government",
                                        6: "Self-employed", 7: "Self-employed"})
                           .fillna("Other / not working"),
        "occupation_group": raw["SOCP"].str[:2].fillna("Not applicable"),
        "race": raw["RAC1P"].map({1: "White", 2: "Black", 6: "Asian", 8: "Other",
                                  9: "Two or more"}).fillna("Other"),
        "nativity": raw["NATIVITY"].map({1: "Native", 2: "Foreign-born"}),
        "disability": raw["DIS"].map({1: "Yes", 2: "No"}),
        "total_income": raw["PINCP"] * adj,
        "wages": raw["WAGP"] * adj,
        "self_emp_income": raw["SEMP"].clip(lower=0) * adj,        # losses recoded to 0
        "investment_income": raw["INTP"].clip(lower=0) * adj,      # losses recoded to 0
        "retirement_income": raw["RETP"] * adj,
    })
    for c in CONFIDENTIAL:
        out[c] = out[c].round(0)
    return out


def add_income_band(df: pd.DataFrame, n_bands: int = N_BANDS) -> pd.DataFrame:
    """Add the derived sensitive class: quintile of total_income (0 = lowest)."""
    out = df.copy()
    out["income_band"] = pd.qcut(out["total_income"].rank(method="first"), n_bands,
                                 labels=False).astype(int)
    return out


def build_samples(n: int = 5000, seed: int = 42, raw_csv: Path = RAW_CSV,
                  save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the raw PUMS file, filter and recode it, and draw two DISJOINT random
    samples of n persons: the *release* sample (the data being protected) and a
    *reference* sample (a stand-in for public population knowledge, used as the
    no-release baseline of the machine-learning attack).
    """
    raw = pd.read_csv(raw_csv, usecols=_USECOLS, low_memory=False)
    raw = raw[(raw["AGEP"] >= 18) & (raw["AGEP"] <= 90) & (raw["PINCP"] > 0)]
    both = _recode(raw.sample(n=2 * n, random_state=seed)).reset_index(drop=True)
    release, reference = both.iloc[:n].reset_index(drop=True), both.iloc[n:].reset_index(drop=True)
    if save:
        release.to_csv(SAMPLE_CSV, index=False)
        reference.to_csv(REFERENCE_CSV, index=False)
    return release, reference


def load_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the committed release and reference samples (rebuilding them if absent)."""
    if SAMPLE_CSV.exists() and REFERENCE_CSV.exists():
        return pd.read_csv(SAMPLE_CSV), pd.read_csv(REFERENCE_CSV)
    return build_samples()
