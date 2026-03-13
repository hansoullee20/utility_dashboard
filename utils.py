"""utils.py — Shared constants and helpers used across all view modules."""
import pandas as pd

BLD_COLOR: dict[str, str] = {
    "A": "#4C72B0",
    "B": "#55A868",
    "C": "#C44E52",
    "D": "#DD8A00",
}


def fmt_won(v: float, signed: bool = False) -> str:
    """Format Korean Won — auto-scale, up to 2 decimals, round up sub-원."""
    import math
    rounded = int(math.copysign(math.ceil(abs(v)), v)) if v != 0 else 0
    sign = ("+" if rounded >= 0 else "") if signed else ""
    abs_r = abs(rounded)
    if abs_r >= 1e8:
        s = f"{rounded / 1e8:,.2f}".rstrip("0").rstrip(".")
        return f"{sign}{s}억"
    elif abs_r >= 1e4:
        s = f"{rounded / 1e4:,.2f}".rstrip("0").rstrip(".")
        return f"{sign}{s}만"
    elif abs_r >= 1e3:
        s = f"{rounded / 1e3:,.2f}".rstrip("0").rstrip(".")
        return f"{sign}{s}천"
    return f"{sign}{rounded:,}원"


# ── Z-score → business-friendly grade ────────────────────────────────────────

_GRADE_THRESHOLDS = [
    (-2.0, "매우낮음", "🟢"),
    (-1.0, "낮음",     "🔵"),
    ( 1.0, "보통",     "⚪"),
    ( 2.0, "높음",     "🟠"),
]
_GRADE_TOP = ("매우높음", "🔴")


def z_to_grade(z: float | None) -> str:
    """Convert Z-score to business-friendly 5-level grade string.

    Returns: '보통', '높음', '매우높음', '낮음', '매우낮음', or '' for NaN.
    """
    if z is None or pd.isna(z):
        return ""
    for threshold, label, _ in _GRADE_THRESHOLDS:
        if z < threshold:
            return label
    return _GRADE_TOP[0]


def z_to_badge(z: float | None) -> str:
    """Convert Z-score to colored emoji + grade label for display.

    Returns: '🟠 높음', '🔴 매우높음', etc.
    """
    if z is None or pd.isna(z):
        return ""
    for threshold, label, emoji in _GRADE_THRESHOLDS:
        if z < threshold:
            return f"{emoji} {label}"
    return f"{_GRADE_TOP[1]} {_GRADE_TOP[0]}"


def z_col_to_grade(s: pd.Series) -> pd.Series:
    """Vectorized Z-score → grade conversion for a DataFrame column."""
    return s.apply(z_to_grade)


def z_col_to_badge(s: pd.Series) -> pd.Series:
    """Vectorized Z-score → badge conversion for a DataFrame column."""
    return s.apply(z_to_badge)


def iqr_upper(s: pd.Series) -> float:
    s = s.dropna()
    s = s[s > 0]
    if len(s) < 4:
        return float("inf")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return float(q3 + 1.5 * (q3 - q1))


def flag_prefix(flags: pd.DataFrame, brand: str) -> str:
    """Return ⛔/⚠/'' prefix based on flag count for a brand."""
    if brand not in flags.index:
        return ""
    val = flags.loc[brand, "플래그 수"]
    n = int(val.iloc[0]) if isinstance(val, pd.Series) else int(val)
    return "⛔ " if n >= 2 else ("⚠ " if n == 1 else "")
