"""LuxAlgo Inversion Fair Value Gap (IFVG) engine, daily bars.

Port of the published Pine v5 indicator:
  Inversion Fair Value Gaps (IFVG) [LuxAlgo]

Green / bullish IFVG = a bearish FVG that price inverted (body traded
through the top). Active zone is then support until the body trades
fully below the bottom.

"Touch" = the as-of daily candle range overlaps a still-valid green IFVG.
Lookback N drops the last N sessions (0 = latest bar) so the scan is
what the filters would have printed that day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BUFFER = 100
ATR_LEN = 200
RSI_LEN = 14
SWING_LEFT = 2
SWING_RIGHT = 2


@dataclass
class FVG:
    left_i: int
    top: float
    right_i: int
    bot: float
    mid: float
    dir: int
    state: int = 0
    x_val_i: int | None = None
    signals: list = field(default_factory=list)


def _rma_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = ATR_LEN) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.full(n, np.nan)
    if n >= length:
        atr[length - 1] = tr[:length].mean()
        alpha = 1.0 / length
        prev = atr[length - 1]
        for i in range(length, n):
            prev = alpha * tr[i] + (1.0 - alpha) * prev
            atr[i] = prev
    return atr


def _wilder_rsi(close: np.ndarray, length: int = RSI_LEN) -> np.ndarray:
    """Wilder RSI. First value at index `length` (14 changes). Matches Pine ta.rsi closely."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < length + 1:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(gain[:length].mean())
    avg_loss = float(loss[:length].mean())
    alpha = 1.0 / length
    if avg_loss == 0:
        out[length] = 100.0
    else:
        out[length] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(length, n - 1):
        avg_gain = alpha * gain[i] + (1.0 - alpha) * avg_gain
        avg_loss = alpha * loss[i] + (1.0 - alpha) * avg_loss
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _atr_threshold(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_multi: float) -> np.ndarray:
    """Pine: nz(ta.atr(200)*atr_multi, cum(high-low)/(bar_index+1))."""
    atr = _rma_atr(high, low, close) * atr_multi
    hl = np.cumsum(high - low) / np.arange(1, len(close) + 1)
    return np.where(np.isnan(atr), hl, atr)


def _fvg_manage(ary: list[FVG], inv_ary: list[FVG], c_bot: float, c_top: float, i: int) -> None:
    if len(ary) >= BUFFER:
        ary.pop(0)
    j = len(ary) - 1
    while j >= 0:
        v = ary[j]
        if v.dir == 1 and c_bot < v.bot:
            v.x_val_i = i
            inv_ary.append(ary.pop(j))
        elif v.dir == -1 and c_top > v.top:
            v.x_val_i = i
            inv_ary.append(ary.pop(j))
        j -= 1


def _inv_manage(
    ary: list[FVG],
    c_bot: float,
    c_top: float,
    close: float,
    high: float,
    low: float,
    close_prev: float,
    i: int,
    wick: bool,
) -> bool:
    fire = False
    if len(ary) >= BUFFER:
        ary.pop(0)
    probe = high if wick else close_prev
    j = len(ary) - 1
    while j >= 0:
        v = ary[j]
        if v.state == 0 and v.dir == 1:
            v.state = 1
            v.dir = -1
        if v.dir == -1 and v.state == 0:
            v.state = 1
            v.dir = 1
        if v.state >= 1:
            v.right_i = i

        if (
            v.dir == -1
            and v.state == 1
            and close < v.bot
            and probe >= v.bot
            and probe < v.top
        ):
            v.signals.append({"i": i, "dir": -1})
            fire = True
        if (
            v.dir == 1
            and v.state == 1
            and close > v.top
            and (low if wick else close_prev) <= v.top
            and (low if wick else close_prev) > v.bot
        ):
            v.signals.append({"i": i, "dir": 1})
            fire = True

        if v.state >= 1 and (
            (v.dir == -1 and c_top > v.top) or (v.dir == 1 and c_bot < v.bot)
        ):
            ary.pop(j)
        j -= 1
    return fire


def last_confirmed_swing_low(
    low: np.ndarray,
    dates: pd.DatetimeIndex,
    last: int,
    left: int = SWING_LEFT,
    right: int = SWING_RIGHT,
) -> dict | None:
    """Most recent 5-bar swing low fully confirmed before `last`.

    Pivot at i: `low[i]` is strictly the lowest of `[i-left, i+right]`.
    Confirmed on bar i+right. Require i+right < last so the last bar is
    not used as confirmation — the swing already existed before this bar.
    """
    end = last - right - 1
    if end < left:
        return None
    for i in range(end, left - 1, -1):
        pivot = low[i]
        ok = True
        for k in range(i - left, i + right + 1):
            if k == i:
                continue
            if low[k] <= pivot:
                ok = False
                break
        if ok:
            return {
                "i": int(i),
                "price": float(pivot),
                "date": dates[i].date().isoformat(),
            }
    return None


def nearest_red_above(red_zones: list, close: float) -> dict | None:
    """Closest still-valid red iFVG at or above close.

    Profit window = % from close up to that zone's bottom (0 if close is
    already inside the red). None if no red sits above price.
    """
    if not close or not red_zones:
        return None
    best = None
    for z in red_zones:
        bot, top = float(z["bot"]), float(z["top"])
        if top < close:
            continue
        if bot <= close:
            return {"pct": 0.0, "bot": bot, "top": top}
        pct = (bot - close) / close * 100.0
        if best is None or pct < best["pct"]:
            best = {"pct": float(pct), "bot": bot, "top": top}
    return best


def _naive_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if getattr(t, "tz", None) is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def lookback_as_of(index, lookback: int):
    """Timestamp of the bar `lookback` sessions before the last. 0 = last bar."""
    if index is None or len(index) == 0:
        return None
    n = int(lookback)
    if n < 0:
        n = 0
    if n >= len(index):
        return None
    return _naive_ts(index[-1 - n])


def universe_as_of(frames: dict, lookback: int):
    """Shared as-of date: SPY calendar, else the longest frame."""
    if not frames:
        return None
    ref = frames.get("SPY")
    if ref is None or getattr(ref, "empty", True):
        ref = max(frames.values(), key=lambda d: 0 if d is None else len(d))
    return lookback_as_of(ref.index, lookback)


def apply_lookback(df: pd.DataFrame, lookback: int = 0, as_of=None) -> pd.DataFrame:
    """Keep bars through N sessions before the last print, or through `as_of`."""
    if df is None or df.empty:
        return df
    if as_of is not None:
        cut = _naive_ts(as_of)
        out = df.copy()
        idx = pd.DatetimeIndex(out.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
            out.index = idx
        return out.loc[out.index <= cut]
    n = int(lookback)
    if n <= 0:
        return df
    if n >= len(df):
        return df.iloc[0:0]
    return df.iloc[:-n]


def scan_ifvg(
    df: pd.DataFrame,
    atr_multi: float = 0.25,
    signal_pref: str = "Close",
    include_ohlc: bool = True,
    lookback: int = 0,
    as_of=None,
) -> dict:
    """Run IFVG on an OHLC daily frame. Index should be dates."""
    df = apply_lookback(df, lookback, as_of=as_of)
    if df is None or df.empty:
        return {"ok": False, "reason": "too few bars"}
    mapping = {}
    for want in ("Open", "High", "Low", "Close"):
        for c in df.columns:
            if str(c).lower() == want.lower():
                mapping[want] = c
                break
    if set(mapping) != {"Open", "High", "Low", "Close"}:
        raise ValueError(f"OHLC columns missing; got {list(df.columns)}")

    df = df.dropna(subset=[mapping["Open"], mapping["High"], mapping["Low"], mapping["Close"]])
    o = df[mapping["Open"]].to_numpy(dtype=float)
    h = df[mapping["High"]].to_numpy(dtype=float)
    l = df[mapping["Low"]].to_numpy(dtype=float)
    c = df[mapping["Close"]].to_numpy(dtype=float)
    n = len(df)
    if n < 5:
        return {"ok": False, "reason": "too few bars"}

    dates = pd.DatetimeIndex(df.index)
    atr = _atr_threshold(h, l, c, atr_multi)
    wick = signal_pref == "Wick"

    bull_fvg: list[FVG] = []
    bear_fvg: list[FVG] = []
    bull_inv: list[FVG] = []  # inverted bull FVGs → red IFVG (dir -1)
    bear_inv: list[FVG] = []  # inverted bear FVGs → green IFVG (dir +1)

    last_bull_signal = False
    last_bear_signal = False

    for i in range(n):
        if i >= 2:
            width_up = abs(l[i] - h[i - 2])
            width_dn = abs(l[i - 2] - h[i])
            fvg_up = (l[i] > h[i - 2]) and (c[i - 1] > h[i - 2])
            fvg_down = (h[i] < l[i - 2]) and (c[i - 1] < l[i - 2])
            if fvg_up and width_up > atr[i]:
                bull_fvg.append(
                    FVG(
                        left_i=i - 1,
                        top=float(l[i]),
                        right_i=i,
                        bot=float(h[i - 2]),
                        mid=float((l[i] + h[i - 2]) / 2),
                        dir=1,
                    )
                )
            if fvg_down and width_dn > atr[i]:
                bear_fvg.append(
                    FVG(
                        left_i=i - 1,
                        top=float(l[i - 2]),
                        right_i=i,
                        bot=float(h[i]),
                        mid=float((h[i] + l[i - 2]) / 2),
                        dir=-1,
                    )
                )

        c_top = max(o[i], c[i])
        c_bot = min(o[i], c[i])
        close_prev = c[i - 1] if i else c[i]
        _fvg_manage(bull_fvg, bull_inv, c_bot, c_top, i)
        _fvg_manage(bear_fvg, bear_inv, c_bot, c_top, i)
        last_bear_signal = _inv_manage(bull_inv, c_bot, c_top, c[i], h[i], l[i], close_prev, i, wick)
        last_bull_signal = _inv_manage(bear_inv, c_bot, c_top, c[i], h[i], l[i], close_prev, i, wick)

    last = n - 1
    last_high, last_low, last_close = h[last], l[last], c[last]
    last_open = o[last]
    last_date = dates[last]
    ssl = last_confirmed_swing_low(l, dates, last)
    sweep = bool(
        ssl is not None and last_low < ssl["price"] and last_close > ssl["price"]
    )
    rsi_arr = _wilder_rsi(c)
    rsi = float(rsi_arr[last]) if not np.isnan(rsi_arr[last]) else None
    rsi_dist_30 = None if rsi is None else float(rsi - 30.0)

    def _zone_dict(v: FVG, kind: str) -> dict:
        sigs = []
        for s in v.signals:
            si = int(s["i"])
            sigs.append(
                {
                    "i": si,
                    "dir": int(s["dir"]),
                    "date": dates[si].date().isoformat(),
                }
            )
        return {
            "kind": kind,
            "top": v.top,
            "bot": v.bot,
            "mid": v.mid,
            "left_i": v.left_i,
            "inv_i": v.x_val_i,
            "formed": dates[v.left_i].date().isoformat(),
            "inverted": dates[v.x_val_i].date().isoformat() if v.x_val_i is not None else None,
            "signals": sigs,
        }

    green_zones = [_zone_dict(v, "green") for v in bear_inv if v.dir == 1 and v.state >= 1]
    red_zones = [_zone_dict(v, "red") for v in bull_inv if v.dir == -1 and v.state >= 1]
    red_above = nearest_red_above(red_zones, float(last_close))

    touches = []
    for v in bear_inv:
        if v.dir != 1 or v.state < 1:
            continue
        overlaps = last_high >= v.bot and last_low <= v.top
        if not overlaps:
            continue
        body_lo, body_hi = min(last_open, last_close), max(last_open, last_close)
        inside_close = v.bot <= last_close <= v.top
        wick_only = overlaps and not (body_lo <= v.top and body_hi >= v.bot)
        age_inv = last - (v.x_val_i if v.x_val_i is not None else v.left_i)
        age_fvg = last - v.left_i
        dist_mid_pct = (last_close - v.mid) / v.mid * 100.0 if v.mid else 0.0
        if v.bot <= last_close <= v.top:
            dist_zone_pct = 0.0
        elif last_close > v.top:
            dist_zone_pct = (last_close - v.top) / last_close * 100.0
        else:
            dist_zone_pct = (v.bot - last_close) / last_close * 100.0
        zone_w_pct = (v.top - v.bot) / last_close * 100.0 if last_close else 0.0
        signaled_today = any(s["i"] == last and s["dir"] == 1 for s in v.signals)
        touches.append(
            {
                "kind": "green_ifvg",
                "top": v.top,
                "bot": v.bot,
                "mid": v.mid,
                "formed": dates[v.left_i].date().isoformat(),
                "inverted": dates[v.x_val_i].date().isoformat() if v.x_val_i is not None else None,
                "age_inv_bars": int(age_inv),
                "age_fvg_bars": int(age_fvg),
                "inside_close": bool(inside_close),
                "wick_only": bool(wick_only),
                "dist_mid_pct": float(dist_mid_pct),
                "dist_zone_pct": float(dist_zone_pct),
                "zone_w_pct": float(zone_w_pct),
                "lux_bull_signal": bool(signaled_today),
                "bear_above_pct": None if red_above is None else float(red_above["pct"]),
                "bear_bot": None if red_above is None else float(red_above["bot"]),
                "bear_top": None if red_above is None else float(red_above["top"]),
                "sweep": bool(sweep),
                "sweep_in_zone": bool(sweep and inside_close),
                "swing_low": None if ssl is None else float(ssl["price"]),
                "swing_date": None if ssl is None else ssl["date"],
                "swing_i": None if ssl is None else int(ssl["i"]),
                "rsi": rsi,
                "rsi_dist_30": rsi_dist_30,
            }
        )

    return {
        "ok": True,
        "last_date": last_date.date().isoformat(),
        "last_close": float(last_close),
        "last_open": float(last_open),
        "last_high": float(last_high),
        "last_low": float(last_low),
        "n_bars": n,
        "n_green_active": len(green_zones),
        "n_red_active": len(red_zones),
        "touches": touches,
        "lux_bull_signal": bool(last_bull_signal),
        "lux_bear_signal": bool(last_bear_signal),
        "green_zones": green_zones,
        "red_zones": red_zones,
        "nearest_red": red_above,
        "sweep": bool(sweep),
        "swing_low": None if ssl is None else float(ssl["price"]),
        "swing_date": None if ssl is None else ssl["date"],
        "swing_i": None if ssl is None else int(ssl["i"]),
        "rsi": rsi,
        "rsi_dist_30": rsi_dist_30,
        "ohlc": (
            df[[mapping["Open"], mapping["High"], mapping["Low"], mapping["Close"]]].copy()
            if include_ohlc
            else None
        ),
    }
