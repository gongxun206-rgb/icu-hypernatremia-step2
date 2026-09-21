from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_restricted_outputs_are_ignored_and_template_has_header_only():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "restricted_data/" in gitignore
    assert "*.csv" in gitignore
    template = ROOT / "config" / "researcher_review_template.csv"
    assert template.read_text(encoding="utf-8").splitlines() == ["admissionid,researcher_final_grade"]
    mimic_template = ROOT / "config" / "mimic_manual_review_template.csv"
    assert mimic_template.read_text(encoding="utf-8").splitlines() == ["stay_id,researcher_final_grade,direct_vs_mixed_semantic_confirmed"]
    if (ROOT / ".git").is_dir():
        ignored = subprocess.run(
            ["git", "check-ignore", "restricted_data/mimic/step2_locked_cohort_key.csv"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert ignored.returncode == 0
