"""utils.py — Shared constants and helpers used across all view modules."""
import numpy as np
import pandas as pd


def display_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Swap normalized ``brand`` with ``brand_raw`` for user-facing display.

    Returns a shallow copy so the caller's DataFrame is not mutated.
    If ``brand_raw`` is absent, returns the original DataFrame unchanged.
    """
    if "brand_raw" not in df.columns:
        return df
    out = df.copy(deep=False)
    out["brand"] = out["brand_raw"]
    return out

BLD_COLOR: dict[str, str] = {
    "A": "#4C72B0",
    "B": "#55A868",
    "C": "#C44E52",
    "D": "#DD8A00",
}

# ── Risk level constants ─────────────────────────────────────────────────────
RISK_DANGER  = "🔴 위험"
RISK_CAUTION = "🟠 주의"
RISK_OBSERVE = "🟡 관찰"
RISK_NORMAL  = "🟢 정상"

RISK_COLOR: dict[str, str] = {
    RISK_DANGER:  "#C44E52",
    RISK_CAUTION: "#DD8A00",
    RISK_OBSERVE: "#F0C040",
    RISK_NORMAL:  "#2ca02c",
}

RISK_PLAIN: dict[str, str] = {
    RISK_DANGER:  "위험",
    RISK_CAUTION: "주의",
    RISK_OBSERVE: "관찰",
    RISK_NORMAL:  "정상",
}

# ── Utility metadata ─────────────────────────────────────────────────────────
UTIL_PREFIXES = ("water", "hwater", "elect", "heat")
UTIL_LABELS: dict[str, str] = {
    "water": "수도", "hwater": "온수", "elect": "전기", "heat": "난방",
}
UTIL_UNITS: dict[str, str] = {
    "water": "m³", "hwater": "m³", "elect": "kWh", "heat": "m³/MWh",
}


# ── Shared formatting helpers ────────────────────────────────────────────────

def fmt_num(val, decimals=1, suffix="", sign=False):
    """Format a numeric value with NaN handling."""
    if val is None:
        return "—"
    try:
        f = float(val)
        if np.isnan(f):
            return "—"
        fmt = f"{{:+,.{decimals}f}}" if sign else f"{{:,.{decimals}f}}"
        return fmt.format(f) + suffix
    except (TypeError, ValueError):
        return "—"


def safe_numeric(val):
    """Convert a single value to float, returning NaN on failure."""
    try:
        return float(pd.to_numeric(val, errors="coerce"))
    except (TypeError, ValueError):
        return float("nan")


def add_per_area_cols(df: pd.DataFrame) -> None:
    """Add per-m² and per-py usage columns to a meter DataFrame in-place."""
    from data import to_numeric_series as _to_num
    size_m2 = _to_num(df["size_m2"]).replace(0, float("nan")) if "size_m2" in df.columns else None
    size_py = _to_num(df["size_py"]).replace(0, float("nan")) if "size_py" in df.columns else None
    for uc, (pm2, ppy) in {
        "water_current":  ("water_usage_per_m2",  "water_usage_per_py"),
        "hwater_current": ("hwater_usage_per_m2", "hwater_usage_per_py"),
        "elect_current":  ("elect_usage_per_m2",  "elect_usage_per_py"),
        "heat_current":   ("heat_usage_per_m2",   "heat_usage_per_py"),
    }.items():
        if uc in df.columns:
            u = _to_num(df[uc])
            if size_m2 is not None:
                df[pm2] = (u / size_m2).round(4)
            if size_py is not None:
                df[ppy] = (u / size_py).round(4)


def load_all_sheets(file_name: str, file_data, all_sheet_keys: list[str],
                    silent: bool = True) -> dict:
    """Load billing/electricity/water/hotwater sheets from an Excel file.

    Args:
        silent: If False, show st.warning on load failure.
    """
    from data import (
        read_billing_sheet, BILLING_SHEET_NAME,
        read_electricity_sheet, ELECTRICITY_SHEET_NAME,
        read_water_sheet, WATER_SHEET_NAME,
        read_hotwater_sheet, HOTWATER_SHEET_NAME,
    )
    loaders = {
        BILLING_SHEET_NAME:     read_billing_sheet,
        ELECTRICITY_SHEET_NAME: read_electricity_sheet,
        WATER_SHEET_NAME:       read_water_sheet,
        HOTWATER_SHEET_NAME:    read_hotwater_sheet,
    }
    results = {}
    for const, loader in loaders.items():
        key = next((k for k in all_sheet_keys if k.strip() == const), None)
        if key is None:
            continue
        try:
            results[const] = loader(file_name, file_data, key)
        except Exception as e:
            if not silent:
                import streamlit as st
                st.warning(f"⚠️ {const} 로드 실패: {e}")
    return results


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


def zscore(s: pd.Series, min_valid: int = 3, decimals: int = 3) -> pd.Series:
    """Robust Z-score: returns NaN when fewer than *min_valid* non-null values."""
    valid = s.dropna()
    if len(valid) < min_valid:
        return pd.Series(np.nan, index=s.index)
    mu, sigma = valid.mean(), valid.std()
    if sigma < 1e-9:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sigma).round(decimals)


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
