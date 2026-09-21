# Data access

This repository contains no patient-level data. MIMIC-IV and AmsterdamUMCdb source data and all patient-level intermediate files remain subject to their respective credentialed-access and data-use agreements. Users must obtain authorization directly from the data custodians and construct local restricted inputs.

For MIMIC-IV, `sql/mimic/00_build_step1_locked_risk_set.sql` can export a required stay-level cohort key only when the user explicitly supplies a local `restricted_output_path`; the script never writes patient-level output to the public repository by default. The cohort key, 27-variable feature dataset, window-level source dataset, frozen prediction files, and manual-review files must remain outside Git. Public code does not include patient-level data or restricted manual-review files.

Do not commit source data, identifiers, row-level manual reviews, model objects, predictions, database dumps, or logs.
