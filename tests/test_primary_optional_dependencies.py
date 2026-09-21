import importlib.util
from pathlib import Path

import pandas as pd

from src.models import run_primary_association as primary


def test_primary_prediction_and_manual_review_dependencies_are_optional():
    baseline = pd.DataFrame({"stay_id": [1, 2]})
    output, prediction_status = primary.load_optional_predictions(baseline, requested=False)
    assert output["predicted_risk"].isna().all()
    assert prediction_status["status"] == "SKIPPED: Step 1 prediction sensitivity not requested"
    assert primary.load_manual_review(None)["status"] == "SKIPPED: restricted manual-review file not supplied"


def test_primary_workflow_initializes_without_restricted_optional_inputs():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_primary_workflow.py"
    spec = importlib.util.spec_from_file_location("run_primary_workflow", script)
    workflow = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(workflow)
    status = workflow.primary_prerequisite_status()
    assert status["optional_sensitivities_required"] is False
    assert status["public_preprocessing_metadata"] is True
