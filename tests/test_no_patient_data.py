from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".parquet", ".feather", ".pkl", ".pickle", ".joblib", ".db", ".sqlite", ".dump", ".backup", ".log"}
ALLOWED_CSV = {
    "config/variable_dictionary.csv",
    "config/researcher_review_template.csv",
    "config/mimic_manual_review_template.csv",
    "config/harmonized/EV_Q02B_A_database_specific_mapping_v1.csv",
    "config/harmonized/EV_Q02B_A_harmonized_common_covariates_v1.csv",
}
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z])[A-Z]:[\\/]")
SECRET_ASSIGNMENT = re.compile(r'''(?i)(password|passwd|api[_-]?key|secret|PGPASSWORD)\s*[:=]\s*['"][^'"]+['"]''')


def test_no_restricted_file_types_or_names():
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path == Path(__file__):
            continue
        rel = path.relative_to(ROOT).as_posix()
        assert path.suffix.lower() not in FORBIDDEN_SUFFIXES, rel
        if path.suffix.lower() == ".csv":
            assert rel in ALLOWED_CSV, rel
        assert path.name not in {".env", "credentials", "secrets"}, rel


def test_no_private_paths_or_assigned_secrets():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts) or path == Path(__file__) or path.suffix.lower() not in {".py", ".sql", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not ABSOLUTE_WINDOWS_PATH.search(text), path
        assert not SECRET_ASSIGNMENT.search(text), path
