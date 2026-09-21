"""Export non-patient preprocessing metadata from a restricted frozen pipeline.

The source joblib is not distributed. This utility produces only variable names,
imputation statistics, encoding rules, and scaling parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pipeline = joblib.load(args.pipeline)
    preprocess = pipeline.named_steps["preprocess"]
    numeric = preprocess.named_transformers_["numeric"]
    categorical = preprocess.named_transformers_["categorical"]
    binary = preprocess.named_transformers_["binary"]
    numeric_names = list(preprocess.transformers_[0][2])
    categorical_names = list(preprocess.transformers_[1][2])
    binary_names = list(preprocess.transformers_[2][2])
    numeric_imputer = numeric.named_steps["imputer"]
    numeric_scaler = numeric.named_steps["scaler"]
    indicator_names = [numeric_names[i] for i in numeric_imputer.indicator_.features_.tolist()]

    metadata = {
        "schema_version": "1.0",
        "source_artifact": args.pipeline.name,
        "contains_patient_level_data": False,
        "numeric": {
            "variables": numeric_names,
            "median_imputation": dict(zip(numeric_names, map(float, numeric_imputer.statistics_))),
            "missing_indicator_variables": indicator_names,
            "standardization_mean": list(map(float, numeric_scaler.mean_)),
            "standardization_scale": list(map(float, numeric_scaler.scale_)),
        },
        "categorical": {
            "variables": categorical_names,
            "most_frequent_imputation": dict(zip(categorical_names, categorical.named_steps["imputer"].statistics_.tolist())),
            "one_hot_categories": {name: values.tolist() for name, values in zip(categorical_names, categorical.named_steps["onehot"].categories_)},
            "association_model_reference_category": {"gender": "F"},
        },
        "binary": {
            "variables": binary_names,
            "most_frequent_imputation": dict(zip(binary_names, map(float, binary.statistics_))),
            "encoding": "0/1 after the frozen boolean normalization rule",
        },
        "transformed_feature_names": list(preprocess.get_feature_names_out()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
