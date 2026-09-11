"""Market cap + average daily dollar volume for options liquidity filters."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
MKTCAP_PKL = CACHE / "market_caps.pkl"
ADV_WINDOW = 20  # sessions
TTL_HOURS = 24


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_mktcap_cache() -> dict:
    if not MKTCAP_PKL.exists():
        return {}
    try:
        raw = pd.read_pickle(MKTCAP_PKL)
    except Exception:
        return {}
    if isinstance(raw, pd.Series):
        raw = raw.to_dict()
    return raw if isinstance(raw, dict) else {}


def _write_mktcap_cache(data: dict) -> None:
    pd.Series(data).to_pickle(MKTCAP_PKL)


def _fresh(rec: dict | None) -> bool:
    if not rec or rec.get("market_cap") is None:
        return False
    fetched = rec.get("fetched")
    if not fetched:
        return False
    try:
        ts = datetime.fromisoformat(str(fetched))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:
        return False
    return (_now() - ts).total_seconds() < TTL_HOURS * 3600


def _fetch_one_mktcap(ticker: str) -> float | None:
    for _ in range(2):
        try:
            t = yf.Ticker(ticker)
            fi = getattr(t, "fast_info", None)
            cap = getattr(fi, "market_cap", None) if fi is not None else None
            if cap is None:
                info = t.info or {}
                cap = info.get("marketCap")
            if cap is not None and float(cap) > 0:
                return float(cap)
        except Exception:
            time.sleep(0.3)
    return None


def ensure_market_caps(tickers: list[str], max_workers: int = 12) -> dict[str, float | None]:
    """Return {ticker: market_cap USD or None}. Cached ~24h."""
    cached = _read_mktcap_cache()
    out: dict[str, float | None] = {}
    need: list[str] = []
    for t in tickers:
        t = str(t).upper()
        rec = cached.get(t)
        if _fresh(rec):
            out[t] = rec.get("market_cap")
        else:
            need.append(t)

    if need:
        fetched_at = _now().isoformat()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_fetch_one_mktcap, t): t for t in need}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    cap = fut.result()
                except Exception:
                    cap = None
                cached[t] = {"market_cap": cap, "fetched": fetched_at}
                out[t] = cap
        _write_mktcap_cache(cached)
    return out


def dollar_adv(
    df: pd.DataFrame,
    window: int = ADV_WINDOW,
    as_of=None,
) -> float | None:
    """20-session average of Close × Volume ending at as_of (or last bar)."""
    if df is None or df.empty:
        return None
    cols = {str(c).lower(): c for c in df.columns}
    if "close" not in cols or "volume" not in cols:
        return None
    frame = df
    if as_of is not None:
        cut = pd.Timestamp(as_of)
        if getattr(cut, "tzinfo", None) is not None:
            cut = cut.tz_localize(None)
        idx = pd.DatetimeIndex(frame.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
            frame = frame.copy()
            frame.index = idx
        frame = frame.loc[frame.index <= cut]
    if frame.empty:
        return None
    close = pd.to_numeric(frame[cols["close"]], errors="coerce")
    vol = pd.to_numeric(frame[cols["volume"]], errors="coerce")
    dollar = (close * vol).dropna()
    if dollar.empty:
        return None
    tail = dollar.iloc[-int(window) :] if len(dollar) >= 1 else dollar
    if tail.empty:
        return None
    return float(tail.mean())


def attach_liquidity(
    hits: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    as_of=None,
    fetch_mktcap: bool = True,
) -> pd.DataFrame:
    """Add market_cap, mktcap_bn, adv_dollar, adv_m columns."""
    if hits is None or hits.empty:
        return hits
    tickers = hits["ticker"].astype(str).unique().tolist()
    caps = ensure_market_caps(tickers) if fetch_mktcap else {}
    rows = {}
    for t in tickers:
        cap = caps.get(t)
        adv = dollar_adv(frames.get(t), as_of=as_of) if t in (frames or {}) else None
        rows[t] = {
            "market_cap": None if cap is None else float(cap),
            "mktcap_bn": None if cap is None else round(float(cap) / 1e9, 2),
            "adv_dollar": None if adv is None else float(adv),
            "adv_m": None if adv is None else round(float(adv) / 1e6, 1),
        }
    extra = pd.DataFrame.from_dict(rows, orient="index")
    extra.index.name = "ticker"
    extra = extra.reset_index()
    drop = [c for c in ("market_cap", "mktcap_bn", "adv_dollar", "adv_m") if c in hits.columns]
    if drop:
        hits = hits.drop(columns=drop)
    return hits.merge(extra, on="ticker", how="left")
