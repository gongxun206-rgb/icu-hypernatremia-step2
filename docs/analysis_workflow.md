# Analysis workflow

1. Reconstruct the locked Step 1 risk set with `sql/mimic/00_build_step1_locked_risk_set.sql`: first adult ICU stay, T0, observed-normal pre-T0 sodium, and pre-T0 hypertonic-saline exclusion. Supply `-v restricted_output_path=...` only for an authorized local restricted cohort-key export.
2. Supply the resulting restricted cohort key outside Git; `sql/mimic/01_build_primary_window_source.sql` then applies T1, management-window event-free, post-T1 observation, interface, and fluid-exposure rules. The frozen Step 1 feature dataset remains a restricted local intermediate because this repository does not reproduce the complete Step 1 prediction-study workflow.
3. Reconstruct direct-fluid and carrier records using frozen semantic and allocation rules.
4. Preserve post-T1 sodium-unascertained status rather than coding it as non-event.
5. Fit MIMIC primary logistic models with the frozen 27-variable framework and co-exposure indicators. `config/frozen_preprocessing_parameters.json` provides the frozen non-patient imputation, encoding, missing-indicator, and scaling parameters; it replaces any primary-path requirement for `lasso_final_development.joblib`.
6. Run restricted cubic spline and prespecified structural/time-to-event sensitivities.
7. Re-express both databases under the harmonized ICU-only endpoint and reduced H1 adjustment set.
8. Compare database-specific results without pooling or heterogeneity testing.
9. Generate only aggregate tables, figures, diagnostics, and validation manifests.

The repository scripts expect authorized local data and restricted intermediate files outside Git. File locations are provided through psql variables and environment variables. Step 1 prediction files and manual-review files are optional sensitivity or QC inputs, not primary-path prerequisites.
