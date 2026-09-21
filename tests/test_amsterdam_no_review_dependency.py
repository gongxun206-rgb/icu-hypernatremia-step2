from argparse import Namespace

from src.models import run_amsterdam_harmonized as amsterdam


def test_amsterdam_import_and_primary_path_do_not_require_review_csv():
    ambiguous_ids, status = amsterdam.optional_ambiguous_review_ids(
        Namespace(run_ambiguous_review_sensitivity=False, review_csv=None)
    )
    assert ambiguous_ids is None
    assert status == "SKIPPED: restricted manual-review sensitivity not requested"
