"""Daily OHLCV via yfinance, parquet-cached for the session day."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)

_OHLCV = {"open", "high", "low", "close", "volume"}


def _cache_path(tag: str) -> Path:
    return CACHE / f"prices_{tag}_{date.today().isoformat()}.parquet"


def _has_volume(wide: pd.DataFrame) -> bool:
    if wide is None or wide.empty or not isinstance(wide.columns, pd.MultiIndex):
        return False
    fields = {str(x).lower() for x in wide.columns.get_level_values(1)}
    return "volume" in fields


def download_daily(tickers: list[str], period: str = "2y", tag: str = "union") -> dict[str, pd.DataFrame]:
    path = _cache_path(tag)
    if path.exists():
        wide = pd.read_parquet(path)
        if _has_volume(wide):
            return _split(wide)
        # Older caches only kept OHLC — drop and re-pull with Volume.
        path.unlink(missing_ok=True)

    tickers = sorted({t.upper() for t in tickers})
    chunks = []
    size = 80
    for i in range(0, len(tickers), size):
        batch = tickers[i : i + size]
        raw = yf.download(
            tickers=batch,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        part = _normalize(raw, batch)
        if not part.empty:
            chunks.append(part)
    if not chunks:
        return {}
    wide = pd.concat(chunks, axis=1)
    wide.to_parquet(path)
    return _split(wide)


def _normalize(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance may be (ticker, field) or (field, ticker)
        lvl0 = [str(x) for x in raw.columns.get_level_values(0)]
        if any(x in {"Open", "High", "Low", "Close", "Adj Close", "Volume"} for x in lvl0):
            raw = raw.swaplevel(0, 1, axis=1)
        raw = raw.sort_index(axis=1)
        keep = [c for c in raw.columns if str(c[1]).lower() in _OHLCV]
        return raw[keep]
    # single ticker
    t = tickers[0]
    cols = [c for c in raw.columns if str(c).lower() in _OHLCV]
    out = raw[cols].copy()
    out.columns = pd.MultiIndex.from_product([[t], [c.title() for c in cols]])
    return out


def _split(wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if wide.empty:
        return out
    tickers = wide.columns.get_level_values(0).unique()
    for t in tickers:
        try:
            sub = wide[t].copy()
        except KeyError:
            continue
        sub.columns = [str(c).title() for c in sub.columns]
        need = [c for c in ("Open", "High", "Low", "Close") if c in sub.columns]
        sub = sub.dropna(subset=need, how="any")
        if len(sub) >= 210:
            out[str(t)] = sub
    return out
