"""utils.py — Shared constants and helpers used across all view modules."""
import pandas as pd

BLD_COLOR: dict[str, str] = {
    "A": "#4C72B0",
    "B": "#55A868",
    "C": "#C44E52",
    "D": "#DD8A00",
}


def fmt_won(v: float, signed: bool = False) -> str:
    """Auto-scale Korean Won: 억원 / 만원 / 원."""
    abs_v = abs(v)
    sign = ("+" if v >= 0 else "") if signed else ""
    if abs_v >= 1e8:
        return f"{sign}{v/1e8:,.1f}억원"
    elif abs_v >= 1e4:
        return f"{sign}{v/1e4:,.1f}만원"
    else:
        return f"{sign}{v:,.0f}원"


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
