"""
End-to-end anonymisation pipeline:

    cleaned data --encode--> distance space X --partition--> labels
                 --aggregate (same labels for every attribute)--> anonymised data

The result object keeps the cleaned original and the anonymised frame aligned
row-by-row. That alignment is needed to *measure* utility and risk, so the
object is internal: anything that leaves the process must go through
``release.build_release`` (which drops the alignment, shuffles rows and runs
the release guard).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import mdav as algorithms
from .config import AnonymizationConfig
from .features import FeatureEncoder, build_encoder


@dataclass
class AnonymizationResult:
    config: AnonymizationConfig
    original: pd.DataFrame               # cleaned original - INTERNAL ONLY
    anonymized: pd.DataFrame             # aligned with ``original`` - INTERNAL ONLY
    labels: np.ndarray | None            # joint partition (None for individual ranking)
    attribute_labels: dict[str, np.ndarray] | None
    encoder: FeatureEncoder
    X: np.ndarray                        # original records in distance space
    runtime_seconds: float

    @property
    def is_joint(self) -> bool:
        """True when one partition is shared by all attributes (MDAV / random)."""
        return self.labels is not None

    @property
    def group_sizes(self) -> np.ndarray | None:
        return None if self.labels is None else algorithms.group_sizes(self.labels)

    @property
    def n_groups(self) -> int | None:
        return None if self.labels is None else int(self.labels.max() + 1)


def anonymize(clean_df: pd.DataFrame, config: AnonymizationConfig) -> AnonymizationResult:
    """Micro-aggregate ``clean_df`` (output of preprocessing.validate_and_clean)."""
    config.validate(len(clean_df))
    numeric = list(config.numeric_attributes)
    categorical = list(config.categorical_attributes)
    missing = [c for c in numeric + categorical if c not in clean_df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")
    if clean_df[numeric + categorical].isna().any().any():
        raise ValueError("Data contains missing values; run validate_and_clean first.")

    original = clean_df.reset_index(drop=True)
    encoder = build_encoder(numeric, categorical, config.categorical_weight,
                            config.log_transform_monetary, config.monetary_attributes)
    X = encoder.fit_transform(original)

    start = time.perf_counter()
    labels = None
    attribute_labels = None
    if config.method == "mdav":
        labels = algorithms.mdav(X, config.k)
        anonymized = algorithms.aggregate(original, labels, numeric, categorical)
    elif config.method == "random":
        labels = algorithms.random_partition(len(original), config.k, config.seed)
        anonymized = algorithms.aggregate(original, labels, numeric, categorical)
    else:  # individual_ranking
        anonymized, attribute_labels = algorithms.aggregate_individual_ranking(
            original, config.k, numeric, categorical)
    runtime = time.perf_counter() - start

    return AnonymizationResult(config, original, anonymized, labels, attribute_labels,
                               encoder, X, runtime)
