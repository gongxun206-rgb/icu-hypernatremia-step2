from datetime import datetime, timedelta

from src.preprocessing.time_windows import harmonized_followup_end, in_left_closed_right_open, landmark_times


def test_landmark_and_interval_boundaries():
    admission = datetime(2026, 1, 1)
    t0, t1 = landmark_times(admission)
    assert t0 == admission + timedelta(hours=24)
    assert t1 == admission + timedelta(hours=36)
    assert in_left_closed_right_open(t0, t0, t1)
    assert not in_left_closed_right_open(t1, t0, t1)


def test_harmonized_followup_earliest_endpoint():
    admission = datetime(2026, 1, 1)
    assert harmonized_followup_end(admission, admission + timedelta(days=8), None) == admission + timedelta(days=7)
    assert harmonized_followup_end(admission, admission + timedelta(days=4), admission + timedelta(days=5)) == admission + timedelta(days=4)
