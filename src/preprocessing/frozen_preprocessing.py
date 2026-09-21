"""Public implementation of the frozen non-patient preprocessing metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class FrozenPreprocessor:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        self.numeric = metadata["numeric"]
        self.categorical = metadata["categorical"]
        self.binary = metadata["binary"]

    @classmethod
    def from_json(cls, path: Path) -> "FrozenPreprocessor":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata["transformed_feature_names"])

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        numeric_names = self.numeric["variables"]
        raw_numeric = frame[numeric_names].apply(pd.to_numeric, errors="coerce")
        missing = raw_numeric[numeric_names].isna()
        medians = pd.Series(self.numeric["median_imputation"])
        imputed = raw_numeric.fillna(medians)
        indicator_names = self.numeric["missing_indicator_variables"]
        indicators = missing[indicator_names].astype(float)
        numeric_plus_indicators = np.column_stack([imputed.to_numpy(dtype=float), indicators.to_numpy(dtype=float)])
        means = np.asarray(self.numeric["standardization_mean"], dtype=float)
        scales = np.asarray(self.numeric["standardization_scale"], dtype=float)
        standardized = (numeric_plus_indicators - means) / scales

        gender = frame["gender"].fillna(self.categorical["most_frequent_imputation"]["gender"]).astype(str)
        categories = self.categorical["one_hot_categories"]["gender"]
        gender_columns = np.column_stack([(gender == category).astype(float).to_numpy() for category in categories])

        binary_names = self.binary["variables"]
        binary_values = frame[binary_names].apply(pd.to_numeric, errors="coerce")
        binary_modes = pd.Series(self.binary["most_frequent_imputation"])
        binary_imputed = binary_values.fillna(binary_modes).to_numpy(dtype=float)

        values = np.column_stack([standardized, gender_columns, binary_imputed])
        return values, self.feature_names
