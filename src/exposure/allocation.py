from __future__ import annotations

from datetime import datetime


def overlap_hours(start: datetime, end: datetime, window_start: datetime, window_end: datetime) -> float:
    """Duration of overlap under left-closed/right-open interval semantics."""
    return max(0.0, (min(end, window_end) - max(start, window_start)).total_seconds() / 3600.0)


def rule_a_rate_overlap(rate_ml_per_hour: float, start: datetime, end: datetime, window_start: datetime, window_end: datetime) -> float:
    """RULE A: rate multiplied by interval overlap."""
    return max(0.0, rate_ml_per_hour) * overlap_hours(start, end, window_start, window_end)


def rule_b_time_prorated_amount(amount_ml: float, start: datetime, end: datetime, window_start: datetime, window_end: datetime) -> float:
    """RULE B: documented amount prorated by overlap/documented duration."""
    duration = (end - start).total_seconds() / 3600.0
    if duration <= 0:
        raise ValueError("Documented duration must be positive for Rule B.")
    return max(0.0, amount_ml) * overlap_hours(start, end, window_start, window_end) / duration


def rule_c_ambiguous() -> None:
    """RULE C has insufficient timing/amount semantics and is not allocated."""
    return None
