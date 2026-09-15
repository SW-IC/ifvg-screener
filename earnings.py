"""Yahoo EPS surprise — same source and rule as Flow B.

Bull: skip unless Yahoo EPS surprise on the last reported print is > 0.
Bear: skip unless that surprise is < 0.
Missing estimate = skip. Stored as a ratio (6.74% → 0.0674).

Reads Flow B's pickle if present; fills holes into this app's cache.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
LOCAL_PKL = CACHE / "earnings_surprise.pkl"
FLOW_B_PKL = (
    Path(__file__).resolve().parent.parent
    / "flow-b-dashboard"
    / "flow_b_cache"
    / "earnings_surprise.pkl"
)


def _as_day(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t.normalize().strftime("%Y-%m-%d")


def _col_by_keyword(df: pd.DataFrame, *needles: str):
    for c in df.columns:
        name = str(c).lower().replace(" ", "")
        if all(n.replace(" ", "") in name for n in needles):
            return c
    return None


def _parse_earnings_surprise(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    surprise_col = _col_by_keyword(df, "surprise")
    est_col = _col_by_keyword(df, "estimate")
    rep_col = _col_by_keyword(df, "reported")
    out = {}
    for ts, row in df.iterrows():
        try:
            day = _as_day(ts)
        except Exception:
            continue
        surprise = row[surprise_col] if surprise_col is not None else float("nan")
        est = row[est_col] if est_col is not None else float("nan")
        rep = row[rep_col] if rep_col is not None else float("nan")
        rec = {
            "surprise": None if pd.isna(surprise) else float(surprise) / 100.0,
            "eps_estimate": None if pd.isna(est) else float(est),
            "eps_reported": None if pd.isna(rep) else float(rep),
        }
        prev = out.get(day)
        if prev is None or (prev.get("surprise") is None and rec["surprise"] is not None):
            out[day] = rec
    return out


def _series_to_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, pd.Series):
        raw = obj.to_dict()
    elif isinstance(obj, dict):
        raw = obj
    else:
        return {}
    out = {}
    for k, v in raw.items():
        out[str(k)] = v if isinstance(v, dict) else {}
    return out


def _read_pkl(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return _series_to_dict(pd.read_pickle(path))
    except Exception:
        return {}


def _write_local(cached: dict) -> None:
    pd.Series(cached).to_pickle(LOCAL_PKL)


def load_surprise_cache() -> dict:
    """Flow B pickle first, then overlay this app's fills."""
    merged = _read_pkl(FLOW_B_PKL)
    local = _read_pkl(LOCAL_PKL)
    for t, recs in local.items():
        if recs:
            merged[t] = recs
    return merged


def fetch_one(ticker: str) -> dict:
    df = None
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).get_earnings_dates(limit=24)
            break
        except Exception:
            time.sleep(0.4)
    if df is None or df.empty:
        return {}
    try:
        return _parse_earnings_surprise(df)
    except Exception:
        return {}


def ensure_surprise(tickers: list[str], cached: dict | None = None, workers: int = 6) -> dict:
    """Return {ticker: {day: rec}}. Fetches names missing from cache."""
    cached = dict(cached or load_surprise_cache())
    tickers = sorted({t.upper() for t in tickers})
    missing = [t for t in tickers if t not in cached or not cached[t]]
    if not missing:
        return {t: cached.get(t, {}) for t in tickers}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_one, t): t for t in missing}
        for fut in as_completed(futs):
            tkr = futs[fut]
            try:
                cached[tkr] = fut.result() or {}
            except Exception:
                cached[tkr] = {}
    _write_local(cached)
    return {t: cached.get(t, {}) for t in tickers}


def last_reported_print(recs: dict, as_of: str) -> dict | None:
    """Most recent print on or before as_of that has a surprise or reported EPS."""
    if not recs:
        return None
    cutoff = str(as_of)[:10]
    best = None
    for day, rec in recs.items():
        if not isinstance(rec, dict) or str(day)[:10] > cutoff:
            continue
        if rec.get("surprise") is None and rec.get("eps_reported") is None:
            continue
        if best is None or str(day)[:10] > best["date"]:
            best = {"date": str(day)[:10], **rec}
    return best


def print_row(recs: dict, as_of: str) -> dict:
    rec = last_reported_print(recs, as_of)
    surprise = None if rec is None else rec.get("surprise")
    beat = surprise is not None and surprise > 0
    miss = surprise is not None and surprise < 0
    return {
        "eps_date": None if rec is None else rec["date"],
        "eps_surprise": surprise,
        "eps_beat": bool(beat),
        "eps_miss": bool(miss),
    }
