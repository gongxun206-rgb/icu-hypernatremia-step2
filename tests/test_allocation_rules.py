from datetime import datetime, timedelta

import pytest

from src.exposure.allocation import rule_a_rate_overlap, rule_b_time_prorated_amount, rule_c_ambiguous


def test_rule_a_rate_overlap():
    t0 = datetime(2026, 1, 1)
    assert rule_a_rate_overlap(100, t0 - timedelta(hours=1), t0 + timedelta(hours=2), t0, t0 + timedelta(hours=1)) == 100


def test_rule_b_prorates_amount():
    t0 = datetime(2026, 1, 1)
    assert rule_b_time_prorated_amount(300, t0 - timedelta(hours=1), t0 + timedelta(hours=2), t0, t0 + timedelta(hours=1)) == 100
    with pytest.raises(ValueError):
        rule_b_time_prorated_amount(100, t0, t0, t0, t0 + timedelta(hours=1))


def test_rule_c_is_not_allocated():
    assert rule_c_ambiguous() is None
