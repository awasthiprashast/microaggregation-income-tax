"""
Distance-space encoding for mixed numeric / categorical microdata.

MDAV needs a metric space. Each record x is mapped to a vector phi(x):

    numeric attribute j      phi_j(x) = (t_j(x_j) - mu_j) / sigma_j
                             where t_j = log1p for monetary attributes
                             (heavy tails) and the identity otherwise;
    categorical attribute c  w * one_hot(x_c)   (one column per level)

and distances are Euclidean in that space. Standardisation makes attributes
measured in rupees and in years commensurable; the log keeps a crore-earner
from dominating every distance; the categorical weight w controls how strongly
MDAV prefers groups that agree on categories (two different levels are
sqrt(2)*w apart, so w = 1/sqrt(2) makes a mismatch cost the same as a 1-sigma
numeric difference).

The encoder is *fitted on the original data* and can then transform released
data into the same space, which is what the linkage attacks need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureEncoder:
    numeric_attributes: tuple[str, ...]
    categorical_attributes: tuple[str, ...]
    categorical_weight: float = 1.0 / np.sqrt(2.0)
    log_attributes: tuple[str, ...] = ()
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)
    levels_: dict[str, list] = field(default_factory=dict)

    def _transform_numeric(self, col: str, values: pd.Series) -> np.ndarray:
        v = values.to_numpy(dtype=float)
        if col in self.log_attributes:
            v = np.log1p(np.maximum(v, 0.0))
        return v

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        for c in self.numeric_attributes:
            v = self._transform_numeric(c, df[c])
            self.means_[c] = float(v.mean())
            std = float(v.std())
            self.stds_[c] = std if std > 0 else 1.0       # constant column -> no scaling
        for c in self.categorical_attributes:
            self.levels_[c] = sorted(df[c].astype(str).unique())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for c in self.numeric_attributes:
            v = self._transform_numeric(c, df[c])
            blocks.append(((v - self.means_[c]) / self.stds_[c])[:, None])
        for c in self.categorical_attributes:
            vals = df[c].astype(str).to_numpy()
            levels = self.levels_[c]
            onehot = (vals[:, None] == np.asarray(levels)[None, :]).astype(float)
            blocks.append(self.categorical_weight * onehot)
        if not blocks:
            return np.empty((len(df), 0))
        return np.hstack(blocks)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features(self) -> int:
        return len(self.numeric_attributes) + sum(len(v) for v in self.levels_.values())


def build_encoder(numeric, categorical, categorical_weight, log_monetary, monetary) -> FeatureEncoder:
    log_attrs = tuple(c for c in numeric if c in monetary) if log_monetary else ()
    return FeatureEncoder(tuple(numeric), tuple(categorical), categorical_weight, log_attrs)


def standardize(df: pd.DataFrame, columns, reference: pd.DataFrame | None = None) -> np.ndarray:
    """Plain z-scores on the raw scale, using the reference frame's moments."""
    ref = df if reference is None else reference
    out = []
    for c in columns:
        mu, sd = ref[c].mean(), ref[c].std(ddof=0)
        out.append(((df[c] - mu) / (sd if sd > 0 else 1.0)).to_numpy(dtype=float))
    return np.column_stack(out) if out else np.empty((len(df), 0))
