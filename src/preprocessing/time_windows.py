from __future__ import annotations

from datetime import datetime, timedelta


def landmark_times(icu_admission: datetime) -> tuple[datetime, datetime]:
    """Return T0=ICU admission+24 h and T1=T0+12 h."""
    t0 = icu_admission + timedelta(hours=24)
    return t0, t0 + timedelta(hours=12)


def in_left_closed_right_open(value: datetime, start: datetime, end: datetime) -> bool:
    return start <= value < end


def harmonized_followup_end(icu_admission: datetime, icu_discharge: datetime, death: datetime | None) -> datetime:
    candidates = [icu_discharge, icu_admission + timedelta(days=7)]
    if death is not None:
        candidates.append(death)
    return min(candidates)
