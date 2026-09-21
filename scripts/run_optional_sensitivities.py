"""Print commands for restricted optional Step 2 sensitivity inputs.

This wrapper deliberately does not run analyses. It separates optional
restricted-input sensitivities from the minimal primary workflow.
"""

from __future__ import annotations


def main() -> None:
    print("Optional MIMIC Step 1-risk sensitivity:")
    print("  set STEP2_STEP1_DEVELOPMENT_PREDICTIONS and STEP2_STEP1_TEMPORAL_PREDICTIONS")
    print("  python -m src.models.run_primary_association --run-step1-risk-sensitivity")
    print()
    print("Optional MIMIC manual-review QC:")
    print("  python -m src.models.run_primary_association --manual-review-csv /restricted/path/reviewer_confirmed.csv")
    print()
    print("Optional Amsterdam ambiguous-review sensitivity:")
    print("  python -m src.models.run_amsterdam_harmonized --run-ambiguous-review-sensitivity --review-csv /restricted/path/researcher_confirmed_review.csv")


if __name__ == "__main__":
    main()
