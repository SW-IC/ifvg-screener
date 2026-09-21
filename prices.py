"""Daily OHLCV via yfinance, parquet-cached for the session day.

`download_live_bar` re-pulls only the last few daily rows so an unfinished
session (market still open) can be merged onto the cached history by
`merge_live_bars`.
"""

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


def _session_date(ts) -> pd.Timestamp:
    """Bar index label as a tz-naive midnight date (US session date)."""
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t.normalize()


def download_live_bar(tickers: list[str], period: str = "5d") -> dict[str, dict]:
    """Latest daily bar per ticker. Today's row is unfinished while the market trades.

    Only ~5 rows per name come back, so this is cheap enough to repeat
    every few minutes. Returns {ticker: {date, Open, High, Low, Close, Volume}}.
    """
    tickers = sorted({t.upper() for t in tickers})
    out: dict[str, dict] = {}
    size = 130
    for i in range(0, len(tickers), size):
        batch = tickers[i : i + size]
        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception:
            continue
        wide = _normalize(raw, batch)
        if wide.empty:
            continue
        for t in wide.columns.get_level_values(0).unique():
            try:
                sub = wide[t].dropna(subset=["Open", "High", "Low", "Close"], how="any")
            except KeyError:
                continue
            if sub.empty:
                continue
            last = sub.iloc[-1]
            row: dict = {"date": _session_date(sub.index[-1])}
            for f in ("Open", "High", "Low", "Close", "Volume"):
                v = last.get(f)
                row[f] = float(v) if v is not None and pd.notna(v) else None
            out[str(t)] = row
    return out


def merge_live_bars(frames: dict[str, pd.DataFrame], live: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Overwrite or append each frame's last bar with its live (unfinished) bar."""
    if not frames or not live:
        return frames
    out: dict[str, pd.DataFrame] = {}
    for t, df in frames.items():
        bar = live.get(str(t))
        if df is None or df.empty or not bar or bar.get("date") is None:
            out[t] = df
            continue
        d = df.copy()
        idx = pd.DatetimeIndex(d.index)
        if idx.tz is not None:
            d.index = idx.tz_convert("America/New_York").tz_localize(None)
            idx = pd.DatetimeIndex(d.index)
        ts = pd.Timestamp(bar["date"])
        if ts < idx[-1]:
            out[t] = d
            continue
        vals = {f: bar[f] for f in d.columns if f in bar and bar.get(f) is not None}
        if ts == idx[-1]:
            for k, v in vals.items():
                d.loc[ts, k] = float(v)
            out[t] = d
        else:
            row = {c: (float(vals[c]) if c in vals else float("nan")) for c in d.columns}
            new = pd.DataFrame([row], index=pd.DatetimeIndex([ts]))
            out[t] = pd.concat([d, new], axis=0)
    return out
