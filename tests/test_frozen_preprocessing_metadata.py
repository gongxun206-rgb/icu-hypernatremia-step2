import json
from pathlib import Path

from src.preprocessing.frozen_preprocessing import FrozenPreprocessor


def test_public_frozen_preprocessing_metadata_has_no_patient_rows():
    path = Path(__file__).resolve().parents[1] / "config" / "frozen_preprocessing_parameters.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["contains_patient_level_data"] is False
    assert len(metadata["numeric"]["variables"]) == 15
    assert len(metadata["numeric"]["missing_indicator_variables"]) == 13
    assert len(metadata["transformed_feature_names"]) == 41
    assert len(FrozenPreprocessor.from_json(path).feature_names) == 41
