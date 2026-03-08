"""utils.py — Shared constants and helpers used across all view modules."""
import pandas as pd

BLD_COLOR: dict[str, str] = {
    "A": "#4C72B0",
    "B": "#55A868",
    "C": "#C44E52",
    "D": "#DD8A00",
}


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
