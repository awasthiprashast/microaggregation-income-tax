"""
Central configuration: attribute roles, valid domains and pipeline defaults.

Attribute roles follow standard statistical-disclosure-control (SDC) terminology:

* Direct identifiers   - identify a person on their own (taxpayer id). They are
                         never released.
* Quasi-identifiers    - individually harmless, but jointly linkable to external
                         sources (age, gender, region, occupation, ...).
* Confidential/        - the values we want to protect (income, deductions, tax).
  sensitive attributes
* Derived sensitive    - attributes computed from confidential ones (tax bracket).
                         They are not released as-is because they would leak the
                         original (un-aggregated) values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Reproducibility / experiment defaults
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
DEFAULT_N_RECORDS: int = 5000
EXPERIMENT_K_VALUES: tuple[int, ...] = (3, 5, 10, 20)
METHODS: tuple[str, ...] = ("mdav", "individual_ranking", "random")
METHOD_LABELS: dict[str, str] = {
    "mdav": "MDAV (multivariate)",
    "individual_ranking": "Individual ranking (univariate)",
    "random": "Random partition (baseline)",
}

# --------------------------------------------------------------------------- #
# Attribute roles for the income/tax schema
# --------------------------------------------------------------------------- #
DIRECT_IDENTIFIERS: tuple[str, ...] = ("taxpayer_id",)

NUMERIC_QUASI_IDENTIFIERS: tuple[str, ...] = ("age", "num_dependents")
CATEGORICAL_QUASI_IDENTIFIERS: tuple[str, ...] = (
    "gender",
    "marital_status",
    "education",
    "occupation",
    "region",
    "city_tier",
)
SENSITIVE_NUMERIC: tuple[str, ...] = (
    "gross_income",
    "deductions",
    "taxable_income",
    "capital_gains",
    "tax_paid",
)
DERIVED_SENSITIVE: tuple[str, ...] = ("tax_bracket",)

# Monetary attributes are heavy-tailed; they are log-transformed in the
# *distance space* (never in the released values) so that a handful of very
# high incomes do not dominate the Euclidean distance.
MONETARY_ATTRIBUTES: tuple[str, ...] = SENSITIVE_NUMERIC

# Attributes that must be integers in the raw data (released values may become
# fractional after averaging - see README "Limitations").
INTEGER_ATTRIBUTES: tuple[str, ...] = ("age", "num_dependents")

# Valid domains used by input validation. None = unbounded on that side.
VALID_RANGES: dict[str, tuple[float | None, float | None]] = {
    "age": (18, 100),
    "num_dependents": (0, 15),
    "gross_income": (0, None),
    "deductions": (0, None),
    "taxable_income": (0, None),
    "capital_gains": (0, None),
    "tax_paid": (0, None),
}

CATEGORY_LEVELS: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male", "Other"),
    "marital_status": ("Divorced", "Married", "Single", "Widowed"),
    "education": ("Secondary", "Higher Secondary", "Graduate", "Postgraduate", "Doctorate"),
    "occupation": (
        "Salaried-Private",
        "Salaried-Government",
        "Self-Employed",
        "Professional",
        "Freelancer",
        "Retired",
    ),
    "region": ("North", "South", "East", "West", "Central", "North-East"),
    "city_tier": ("Tier-1", "Tier-2", "Tier-3"),
}

# A row with more than this share of its analysis attributes missing is dropped
# rather than imputed (imputing most of a record would fabricate a person).
MAX_MISSING_SHARE_PER_ROW: float = 0.5


@dataclass(frozen=True)
class AnonymizationConfig:
    """
    Parameters of one anonymisation run.

    Attributes
    ----------
    k : int
        Minimum group size. Every released record is identical (on all released
        attributes) to at least k-1 others when a multivariate method is used.
    method : str
        "mdav", "individual_ranking" or "random".
    numeric_attributes / categorical_attributes : tuple[str, ...]
        Attributes that are micro-aggregated and therefore released. Anything
        not listed here is NOT released.
    categorical_weight : float
        Scale applied to one-hot columns in the distance space. With weight
        1/sqrt(2) a category mismatch adds exactly 1.0 to the squared distance,
        i.e. the same as a one-standard-deviation difference on a numeric attribute.
    log_transform_monetary : bool
        Use log1p() for monetary attributes in the distance space.
    seed : int
        Seed for every random step (random baseline, release shuffling, attacks).
    """

    k: int = 5
    method: str = "mdav"
    numeric_attributes: tuple[str, ...] = NUMERIC_QUASI_IDENTIFIERS + SENSITIVE_NUMERIC
    categorical_attributes: tuple[str, ...] = CATEGORICAL_QUASI_IDENTIFIERS
    categorical_weight: float = 1.0 / math.sqrt(2.0)
    log_transform_monetary: bool = True
    seed: int = RANDOM_SEED
    monetary_attributes: tuple[str, ...] = field(default=MONETARY_ATTRIBUTES)

    def validate(self, n_records: int) -> None:
        """Raise ValueError if the configuration cannot be applied to n_records rows."""
        if not isinstance(self.k, int) or self.k < 2:
            raise ValueError(f"k must be an integer >= 2 (got {self.k!r}).")
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}; choose one of {METHODS}.")
        if n_records < self.k:
            raise ValueError(
                f"Cannot build groups of size k={self.k} from only {n_records} records."
            )
        if not self.numeric_attributes and not self.categorical_attributes:
            raise ValueError("Select at least one attribute to micro-aggregate.")
        overlap = set(self.numeric_attributes) & set(self.categorical_attributes)
        if overlap:
            raise ValueError(f"Attributes listed as both numeric and categorical: {overlap}")
        if self.categorical_weight < 0:
            raise ValueError("categorical_weight must be non-negative.")

    @property
    def released_attributes(self) -> tuple[str, ...]:
        return tuple(self.numeric_attributes) + tuple(self.categorical_attributes)
