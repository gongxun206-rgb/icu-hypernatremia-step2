"""Check the minimal MIMIC Step 2 primary-workflow prerequisites.

This orchestrator does not fit a model or replace the frozen analysis scripts.
It makes the authorized database-to-restricted-input execution order explicit.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_ROOT = Path(os.environ.get("STEP2_RESTRICTED_ROOT", REPO_ROOT / "restricted_data"))
PRIMARY_FILES = [
    RESTRICTED_ROOT / "mimic" / "step2_locked_cohort_key.csv",
    RESTRICTED_ROOT / "mimic" / "primary_cohort_model_dataset_v1.0.csv",
    RESTRICTED_ROOT / "mimic" / "step2_new_q05_window_level_analysis_source_v1.csv",
]


def primary_prerequisite_status() -> dict[str, object]:
    missing = [str(path) for path in PRIMARY_FILES if not path.is_file()]
    return {
        "public_preprocessing_metadata": (REPO_ROOT / "config" / "frozen_preprocessing_parameters.json").is_file(),
        "restricted_primary_inputs_present": not missing,
        "missing_restricted_inputs": missing,
        "optional_sensitivities_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-commands", action="store_true")
    args = parser.parse_args()
    status = primary_prerequisite_status()
    print(status)
    if args.show_commands:
        print("1. Run sql/mimic/00_build_step1_locked_risk_set.sql with -v restricted_output_path=...")
        print("2. Run sql/mimic/01_build_primary_window_source.sql with -v locked_cohort_csv=... -v window_source_csv=...")
        print("3. Run python -m src.models.run_primary_association")
    if not status["restricted_primary_inputs_present"]:
        print("Primary model not started: restricted primary inputs are absent. Optional sensitivity files are not required.")


if __name__ == "__main__":
    main()
