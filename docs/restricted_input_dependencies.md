# Restricted input dependencies

This repository distributes no patient-level data. All files below must stay in an authorized local directory outside Git.

| Classification | Filename or input | Patient-level data | Public | Why it is restricted | Produced by | Required for | Public substitute | Can be removed from primary path? |
|---|---|---|---|---|---|---|---|---|
| A. Primary | Authorized MIMIC-IV v3.1 database | Yes | No | Credentialed source database | Data custodian | MIMIC extraction | No | No |
| A. Primary | `step2_locked_cohort_key.csv` | Yes | No | Stay-level identifiers and locked eligibility outcome | `sql/mimic/00_build_step1_locked_risk_set.sql` optional restricted export | Step 2 MIMIC SQL | SQL and output schema only | No |
| A. Primary | `primary_cohort_model_dataset_v1.0.csv` | Yes | No | Frozen 27-variable stay-level baseline feature dataset | Authorized Step 1 feature workflow | MIMIC primary adjustment | Variable dictionary and preprocessing metadata only | No |
| A. Primary | `step2_new_q05_window_level_analysis_source_v1.csv` | Yes | No | Stay-level T0/T1 exposure and outcome-process source | `sql/mimic/01_build_primary_window_source.sql` | MIMIC primary association | SQL and output schema only | No |
| A. Primary | Authorized AmsterdamUMCdb database | Yes | No | Credentialed source database | Data custodian | Amsterdam primary analysis | No | No |
| Public metadata | `config/frozen_preprocessing_parameters.json` | No | Yes | Frozen medians, scaling, missing indicators, and encoding rules only | `scripts/export_frozen_preprocessing_metadata.py` using a restricted frozen pipeline | MIMIC primary adjustment | Itself | Not applicable |
| B. Sensitivity only | `lasso_development_nested_oof_predictions.csv` and `lasso_temporal_test_predictions.csv` | Yes | No | Frozen Step 1 predicted risks | Locked Step 1 model workflow | Step 1-risk compact sensitivity and interaction only | No | Yes |
| B. Sensitivity only | Amsterdam confirmed review CSV | Yes | No | Admission-level ambiguous-review grades | Researcher review | Amsterdam ambiguous-review sensitivity only | Header-only template | Yes |
| C. QC / semantic review only | MIMIC confirmed 60-case review CSV | Yes | No | Stay-level review grades and semantic confirmation | Researcher review | Manual-review QC only | Header-only template and aggregate summary | Yes |
| C. QC / semantic review only | Structural sensitivity source files | Yes | No | Stay-level medication and measurement timing | Q07 extraction SQL | Structural/QC analyses only | SQL and aggregate outputs only | Yes |
| D. Legacy / unused | `lasso_final_development.joblib` | No patient rows, but non-public artifact | No | Replaced in the public primary path by exported metadata | Locked Step 1 model workflow | Metadata export only | `config/frozen_preprocessing_parameters.json` | Yes |

The empty [`researcher_review_template.csv`](../config/researcher_review_template.csv) and [`mimic_manual_review_template.csv`](../config/mimic_manual_review_template.csv) contain headers only. The actual restricted manual-review files are not distributed.

Optional inputs never block the primary MIMIC or Amsterdam workflows. When an optional analysis is requested without its restricted file, the corresponding output records `SKIPPED`.
