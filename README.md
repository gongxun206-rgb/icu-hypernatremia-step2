# ICU hypernatremia Step 2 analysis code

Code for **Documented early post-landmark direct saline exposure and subsequent recorded moderate-to-severe ICU-acquired hypernatremia: a two-database retrospective cohort study**.

## Purpose

This repository documents the frozen cohort, exposure reconstruction, association models, structural sensitivities, harmonized external replication, and reporting workflow used in the manuscript.

The minimal Step 1 components included in this repository are provided only to reconstruct the frozen risk-set and preprocessing inputs required for the Step 2 analysis. This repository does not reproduce the complete Step 1 prediction-study workflow.

## Data sources and access

The study used MIMIC-IV v3.1 and AmsterdamUMCdb. **No patient-level data are included.** Users must independently obtain authorized access and comply with each database's data-use agreement. Restricted source data, patient identifiers, intermediate CSV files, predictions, model objects, and manual-review rows must remain outside Git.

## Workflow

1. Step 1 risk-set reconstruction followed by Step 2 T0/T1 cohort construction.
2. Frozen 27-variable pre-T0 framework.
3. Direct-fluid and medication-carrier reconstruction with Rules A/B/C.
4. Outcome ascertainment and observation-process audit.
5. MIMIC primary logistic/RCS analyses and prespecified structural sensitivities.
6. Harmonized ICU-only replication in MIMIC and AmsterdamUMCdb.
7. Database-specific comparison without pooling.
8. Aggregate tables, figures, and validation outputs.

See `docs/analysis_workflow.md`, `docs/semantic_rules.md`, and `docs/harmonization.md`.

## Minimal primary workflow

### MIMIC-IV

The minimal MIMIC primary path requires authorized MIMIC-IV v3.1 access, this public repository, and a restricted local output directory outside the repository. Run the following in order:

```text
sql/mimic/00_build_step1_locked_risk_set.sql
sql/mimic/01_build_primary_window_source.sql
python -m src.models.run_primary_association
```

The first SQL script writes the stay-level cohort key only when an explicit `restricted_output_path` psql variable is supplied. The primary association script requires the restricted cohort key, frozen 27-variable feature dataset, and Step 2 window-source dataset described in `docs/restricted_input_dependencies.md`. It uses the public non-patient preprocessing metadata in `config/frozen_preprocessing_parameters.json`; it does not require a binary Step 1 model object.

For a prerequisite check and command summary, run:

```bash
python scripts/run_primary_workflow.py --show-commands
```

### AmsterdamUMCdb

The Amsterdam harmonized primary analysis requires authorized AmsterdamUMCdb access and does not require a researcher manual-review CSV:

```bash
python -m src.models.run_amsterdam_harmonized
```

The ambiguous-review sensitivity is optional and requires an explicit flag and restricted local review file:

```bash
python -m src.models.run_amsterdam_harmonized --run-ambiguous-review-sensitivity --review-csv /restricted/path/researcher_confirmed_review.csv
```

`config/researcher_review_template.csv` and `config/mimic_manual_review_template.csv` contain no patient-level data. The actual restricted manual-review files are not distributed.

## Reference results

- MIMIC primary: n=29,215; events=661; NaCl OR 1.167 (95% CI 1.092-1.246); LR OR 1.008 (0.934-1.089); D5W OR 1.014 (0.891-1.153).
- Harmonized MIMIC: n=18,119; events=500; NaCl OR 1.207 (1.131-1.287).
- AmsterdamUMCdb: n=2,456; events=282; NaCl OR 1.406 (1.244-1.589).

These are observational database-specific estimates, not causal treatment effects. No pooled estimate or heterogeneity test was performed.

## Environment

Exact historical Python, principal-package, and MIMIC PostgreSQL versions were not preserved. The unpinned requirements file is a compatibility template rather than the historical environment.

## Testing

```bash
python -m pytest -q
```

The tests cover time-window boundaries, Rules A/B/C, and repository safety. They do not substitute for authorized database execution.

## Citation and archive

Release target: `v1.0.0`; Git tag target: `manuscript-submission-v1`. The GitHub repository URL and Zenodo version DOI will be added only after author review and public release.

## License

Project-authored code is released under the MIT License. No third-party AmsterdamUMCdb or eICU repository code is copied into this repository.
