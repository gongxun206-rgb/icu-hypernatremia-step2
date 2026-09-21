# Running the workflow

1. Copy `config/config.example.yaml` outside the repository and fill non-secret local connection settings.
2. Set `STEP2_RESTRICTED_ROOT` and `STEP2_OUTPUT_ROOT` to directories outside tracked source files.
3. For Amsterdam, set `AMSTERDAM_PSQL`, `AMSTERDAM_DB_HOST`, `AMSTERDAM_DB_PORT`, `AMSTERDAM_DB_USER`, and `AMSTERDAM_DB_NAME`. Supply the database password through the operating system or PostgreSQL password file, never source code. A manual-review CSV is optional and is used only with `--run-ambiguous-review-sensitivity --review-csv ...`.
4. Run SQL with the required `-v` psql path variables documented in each script.
5. Run Python scripts in primary, sensitivity, harmonized, and reporting order.
6. Run `python -m pytest -q` before release.
