"""Green iFVG daily screener — S&P 500 ∪ Nasdaq-100.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ifvg as _ifvg

importlib.reload(_ifvg)
from earnings import ensure_surprise, load_surprise_cache, print_row
from ifvg import scan_ifvg, universe_as_of
from liquidity import attach_liquidity
from prices import download_daily
from universe import load_universe

st.set_page_config(page_title="Green iFVG screener", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; max-width: 1400px;}
      h1 {font-size: 1.6rem !important; letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Green iFVG screener")
st.caption(
    "Daily candles overlapping an active **bullish inversion FVG** "
    "(LuxAlgo IFVG: inverted bearish FVG, still valid). "
    "Universe = S&P 500 ∪ Nasdaq-100."
)

with st.sidebar:
    st.header("Scan")
    universe_choice = st.selectbox(
        "Universe",
        ["S&P 500 ∪ Nasdaq-100", "S&P 500", "Nasdaq-100"],
        index=0,
    )
    atr_multi = st.number_input("ATR multiplier", min_value=0.0, max_value=2.0, value=0.25, step=0.25)
    period = st.selectbox("History", ["2y", "1y", "5y"], index=0)
    lookback = st.number_input(
        "Lookback (sessions)",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
        help=(
            "Re-run the same filters as of N sessions ago. "
            "0 = latest bar. 1 = previous session (default). "
            "Uses the SPY calendar so every name shares one as-of date. Hit Scan."
        ),
    )
    show_lux = st.checkbox(
        "LuxAlgo retest signals (▲/▼)",
        value=False,
        help="Off by default. The triangle is LuxAlgo’s retest print (close back through the zone), not a touch. Zones and the scan do not depend on it.",
    )
    if show_lux:
        signal_pref = st.selectbox("Signal preference (LuxAlgo)", ["Close", "Wick"], index=0)
        only_signal = st.checkbox("LuxAlgo bullish signal today only", value=False)
    else:
        signal_pref = "Close"
        only_signal = False
    only_sweep = st.checkbox(
        "Liquidity sweep (5-bar swing low)",
        value=False,
        help="Last bar Low undercuts the most recent confirmed 5-bar swing low, Close reclaims that swing, and Close sits inside the green iFVG.",
    )
    require_eps_beat = st.checkbox(
        "Require previous EPS surprise > 0",
        value=False,
        help=(
            "Last reported Yahoo EPS surprise (the print already out) must be > 0. "
            "Missing estimate = skip. Same rule as Flow B. EPS vs consensus, not GAAP."
        ),
    )
    max_gap = st.slider(
        "Max % gap from zone to close",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="0 = close must sit inside the zone. 2 = close can sit up to 2% outside the IFVG and still count as a touch (wicks).",
    )
    min_bear = st.slider(
        "Min % to nearest red iFVG",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="Profit window: keep names whose next bearish iFVG (resistance above close) is at least this far. No red above = passes. 0 = off.",
    )
    max_rsi_dist = st.slider(
        "Max distance from RSI 30",
        min_value=0,
        max_value=70,
        value=70,
        step=1,
        help="Wilder RSI(14). Distance = RSI − 30. Keep if RSI ≤ 30 + this. 0 = RSI ≤ 30. 70 = off (RSI ≤ 100). RSI below 30 always passes.",
    )
    st.subheader("Liquidity (options)")
    min_mktcap_bn = st.number_input(
        "Min market cap ($B)",
        min_value=0.0,
        max_value=500.0,
        value=10.0,
        step=1.0,
        help="Yahoo market cap. Keep names ≥ this. 0 = off. Default $10B for listed-options liquidity.",
    )
    min_adv_m = st.number_input(
        "Min 20d dollar ADV ($M)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=5.0,
        help="Average of Close × Volume over the last 20 sessions (as-of bar). Keep ≥ this. 0 = off. Default $50M.",
    )
    nearest_only = st.checkbox("One row per ticker (nearest zone)", value=True)
    run = st.button("Scan", type="primary", use_container_width=True)

UNI_KEY = {
    "S&P 500 ∪ Nasdaq-100": "union",
    "S&P 500": "sp500",
    "Nasdaq-100": "ndx100",
}


@st.cache_data(show_spinner="Loading index constituents…", ttl=86400)
def _universe():
    return load_universe()


@st.cache_data(show_spinner="Downloading daily bars…", ttl=60 * 60 * 6)
def _prices(tickers: tuple[str, ...], period: str, tag: str):
    # _v2 = OHLCV (Volume kept for dollar ADV). Bumps Streamlit cache + parquet tag.
    return download_daily(list(tickers), period=period, tag=f"{tag}_{period}_v2")


GREEN = "rgba(8,153,129,0.20)"
RED = "rgba(242,54,69,0.20)"
GREEN_SOLID = "#089981"
RED_SOLID = "#f23645"
MID = "#787b86"
# LuxAlgo send_it: box from formed→invert in original FVG color, invert→now in
# inverted color; last `disp_num` of each array; dashed mid; ▲/▼ on retests.
EXTEND_BARS = 10

TAG_NONE = "○"
TAG_COLORS = ("green", "yellow", "red")
TAG_SYM = {"": TAG_NONE, "green": "🟢", "yellow": "🟡", "red": "🔴"}
SYM_TAG = {TAG_NONE: "", "🟢": "green", "🟡": "yellow", "🔴": "red"}
TAG_PATH = Path(__file__).resolve().parent / "cache" / "ticker_tags.json"


def _load_tags() -> dict[str, str]:
    if not TAG_PATH.exists():
        return {}
    try:
        raw = json.loads(TAG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, str] = {}
    for k, v in (raw or {}).items():
        if v in TAG_COLORS:
            out[str(k)] = v
    return out


def _save_tags(tags: dict[str, str]) -> None:
    TAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TAG_PATH.write_text(json.dumps(tags, indent=2, sort_keys=True), encoding="utf-8")


def _tag_of(ticker: str) -> str:
    t = (st.session_state.get("ticker_tags") or {}).get(str(ticker), "")
    return t if t in TAG_COLORS else ""


def _sync_tags_from_editor(edited: pd.DataFrame) -> None:
    if edited is None or edited.empty or "ticker" not in edited.columns or "tag" not in edited.columns:
        return
    tags = dict(st.session_state.ticker_tags)
    changed = False
    for t, sym in zip(edited["ticker"].astype(str), edited["tag"].astype(str)):
        new = SYM_TAG.get(sym, "")
        old = tags.get(t, "")
        if new == old:
            continue
        changed = True
        if new:
            tags[t] = new
        else:
            tags.pop(t, None)
    if changed:
        st.session_state.ticker_tags = tags
        _save_tags(tags)


if "ticker_tags" not in st.session_state:
    st.session_state.ticker_tags = _load_tags()


def _last_n(zones: list, n: int) -> list:
    if not zones or n <= 0:
        return []
    return zones[-n:]


def _bar_ts(index, i: int):
    i = max(0, min(int(i), len(index) - 1))
    return index[i]


def _chart(result: dict, ticker: str, show_last: int = 5, show_signals: bool = False) -> go.Figure:
    ohlc_full = result["ohlc"]
    dates = ohlc_full.index
    greens = _last_n(result.get("green_zones") or [], show_last)
    reds = _last_n(result.get("red_zones") or [], show_last)
    shown = [("green", z) for z in greens] + [("red", z) for z in reds]

    n = len(ohlc_full)
    start_i = max(0, n - 180)
    if shown:
        earliest = min(int(z["left_i"]) for _, z in shown)
        start_i = min(start_i, max(0, earliest))
    ohlc = ohlc_full.iloc[start_i:]
    last_ts = dates[-1]
    if n >= 2:
        delta = dates[-1] - dates[-2]
        if getattr(delta, "value", 0) <= 0:
            delta = pd.Timedelta(days=1)
        right_ts = last_ts + delta * EXTEND_BARS
    else:
        right_ts = last_ts + pd.Timedelta(days=14)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlc.index,
                open=ohlc["Open"],
                high=ohlc["High"],
                low=ohlc["Low"],
                close=ohlc["Close"],
                name=ticker,
                increasing_line_color=GREEN_SOLID,
                decreasing_line_color=RED_SOLID,
            )
        ]
    )

    def _rect(x0, x1, y0, y1, fill):
        if x0 is None or x1 is None or x1 <= x0:
            return
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=fill,
            line=dict(width=0),
            layer="below",
        )

    up_x, up_y, dn_x, dn_y = [], [], [], []
    vis0, vis1 = ohlc.index[0], right_ts

    for kind, z in shown:
        # After invert: green iFVG was a bear FVG (red), red iFVG was a bull FVG
        # (green). Matches Pine send_it: orig color then flipped color.
        orig = RED if kind == "green" else GREEN
        inv = GREEN if kind == "green" else RED
        left = _bar_ts(dates, z["left_i"])
        inv_i = z.get("inv_i")
        inv_ts = _bar_ts(dates, inv_i) if inv_i is not None else left
        _rect(max(left, vis0), min(inv_ts, vis1), z["bot"], z["top"], orig)
        _rect(max(inv_ts, vis0), vis1, z["bot"], z["top"], inv)
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=max(left, vis0),
            x1=vis1,
            y0=z["mid"],
            y1=z["mid"],
            line=dict(color=MID, dash="dash", width=1),
            layer="below",
        )
        if show_signals:
            for s in z.get("signals") or []:
                si = int(s["i"])
                if si < start_i or si >= n:
                    continue
                ts = dates[si]
                if s["dir"] == 1:
                    up_x.append(ts)
                    up_y.append(z["bot"])
                else:
                    dn_x.append(ts)
                    dn_y.append(z["top"])

    if up_x:
        fig.add_trace(
            go.Scatter(
                x=up_x,
                y=up_y,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=GREEN_SOLID, line=dict(width=0)),
                name="▲ bull",
                hoverinfo="skip",
            )
        )
    if dn_x:
        fig.add_trace(
            go.Scatter(
                x=dn_x,
                y=dn_y,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color=RED_SOLID, line=dict(width=0)),
                name="▼ bear",
                hoverinfo="skip",
            )
        )

    ssl = result.get("swing_low")
    si = result.get("swing_i")
    if ssl is not None:
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=vis0,
            x1=vis1,
            y0=ssl,
            y1=ssl,
            line=dict(color="#f6c177", dash="dot", width=1),
            layer="below",
        )
        if si is not None and 0 <= int(si) < n:
            fig.add_trace(
                go.Scatter(
                    x=[dates[int(si)]],
                    y=[ssl],
                    mode="markers",
                    marker=dict(symbol="diamond", size=8, color="#f6c177", line=dict(width=0)),
                    name="5-bar swing low",
                    hoverinfo="skip",
                )
            )

    n_g = len(result.get("green_zones") or [])
    n_r = len(result.get("red_zones") or [])
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=False,
        title=(
            f"{ticker}  ·  last {result['last_date']}  ·  {result['last_close']:.2f}"
            f"  ·  show last {show_last}  ({len(greens)}/{n_g} green, {len(reds)}/{n_r} red)"
        ),
    )
    fig.update_xaxes(range=[ohlc.index[0], right_ts])
    return fig


def _scan_kwargs(meta: dict | None = None, **extra) -> dict:
    meta = meta or {}
    out = {
        "atr_multi": float(extra.get("atr_multi", meta.get("atr_multi", 0.25))),
        "signal_pref": extra.get("signal_pref", meta.get("signal_pref", "Close")),
        "lookback": int(extra.get("lookback", meta.get("lookback", 0))),
        "as_of": extra.get("as_of", meta.get("as_of")),
    }
    return out


def _run_scan(
    frames: dict[str, pd.DataFrame],
    members: dict,
    atr_multi: float,
    signal_pref: str,
    lookback: int = 1,
) -> pd.DataFrame:
    rows = []
    as_of = universe_as_of(frames, int(lookback))
    for t, df in frames.items():
        try:
            res = scan_ifvg(
                df,
                atr_multi=atr_multi,
                signal_pref=signal_pref,
                include_ohlc=False,
                lookback=int(lookback),
                as_of=as_of,
            )
        except Exception:
            continue
        if not res.get("ok") or not res.get("touches"):
            continue
        for z in res["touches"]:
            rows.append(
                {
                    "ticker": t,
                    "sp500": t in members["sp500"],
                    "ndx100": t in members["ndx100"],
                    "date": res["last_date"],
                    "close": res["last_close"],
                    "ifvg_top": z["top"],
                    "ifvg_bot": z["bot"],
                    "ifvg_mid": z["mid"],
                    "inside_close": z["inside_close"],
                    "wick_only": z["wick_only"],
                    "dist_mid_%": round(z["dist_mid_pct"], 2),
                    "dist_zone_%": round(z["dist_zone_pct"], 2),
                    "zone_w_%": round(z["zone_w_pct"], 2),
                    "inverted": z["inverted"],
                    "formed": z["formed"],
                    "age_inv": z["age_inv_bars"],
                    "lux_signal": z["lux_bull_signal"],
                    "bear_above_%": None if z.get("bear_above_pct") is None else round(z["bear_above_pct"], 2),
                    "bear_bot": z.get("bear_bot"),
                    "bear_top": z.get("bear_top"),
                    "sweep": z.get("sweep", False),
                    "sweep_in_zone": z.get("sweep_in_zone", False),
                    "swing_low": z.get("swing_low"),
                    "swing_date": z.get("swing_date"),
                    "rsi": None if z.get("rsi") is None else round(z["rsi"], 1),
                    "rsi_dist_30": None if z.get("rsi_dist_30") is None else round(z["rsi_dist_30"], 1),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    cols = ["inside_close", "ticker"]
    if "sweep_in_zone" in out.columns:
        cols = ["sweep_in_zone", "inside_close", "ticker"]
    return out.sort_values(cols, ascending=[False] * (len(cols) - 1) + [True])


def _attach_eps(hits: pd.DataFrame, fetch_missing: bool) -> pd.DataFrame:
    if hits is None or hits.empty:
        return hits
    tickers = hits["ticker"].astype(str).unique().tolist()
    as_of = str(hits["date"].iloc[0])
    if fetch_missing:
        recs = ensure_surprise(tickers)
    else:
        cached = load_surprise_cache()
        recs = {t: cached.get(t, {}) for t in tickers}
    rows = {t: print_row(recs.get(t) or {}, as_of) for t in tickers}
    extra = pd.DataFrame.from_dict(rows, orient="index")
    extra.index.name = "ticker"
    extra = extra.reset_index()
    drop = [c for c in ("eps_date", "eps_surprise", "eps_beat") if c in hits.columns]
    if drop:
        hits = hits.drop(columns=drop)
    return hits.merge(extra, on="ticker", how="left")


if run:
    uni = _universe()
    key = UNI_KEY[universe_choice]
    tickers = tuple(uni[key])
    t0 = time.time()
    frames = _prices(tickers, period, key)
    members = {"sp500": set(uni["sp500"]), "ndx100": set(uni["ndx100"])}
    hits = _run_scan(frames, members, float(atr_multi), signal_pref, lookback=int(lookback))
    hits = _attach_eps(hits, fetch_missing=True)
    as_of = universe_as_of(frames, int(lookback))
    hits = attach_liquidity(hits, frames, as_of=as_of, fetch_mktcap=True)
    elapsed = time.time() - t0
    st.session_state["hits"] = hits
    st.session_state["frames"] = frames
    st.session_state["scan_meta"] = {
        "n_uni": len(tickers),
        "n_priced": len(frames),
        "elapsed": elapsed,
        "atr_multi": atr_multi,
        "signal_pref": signal_pref,
        "lookback": int(lookback),
        "as_of": None if as_of is None else pd.Timestamp(as_of).date().isoformat(),
        "uni": uni,
    }

hits = st.session_state.get("hits")
meta = st.session_state.get("scan_meta")

if hits is not None and not hits.empty and "sweep" not in hits.columns:
    frames = st.session_state.get("frames") or {}
    cache = {}
    for t in hits["ticker"].unique():
        if t not in frames:
            continue
        try:
            r = scan_ifvg(frames[t], include_ohlc=False, **_scan_kwargs(meta))
        except Exception:
            continue
        cache[t] = {
            "sweep": bool(r.get("sweep")),
            "swing_low": r.get("swing_low"),
            "swing_date": r.get("swing_date"),
        }
    if cache:
        extra = pd.DataFrame.from_dict(cache, orient="index")
        extra.index.name = "ticker"
        extra = extra.reset_index()
        hits = hits.merge(extra, on="ticker", how="left")
        if "inside_close" in hits.columns:
            hits["sweep_in_zone"] = hits["sweep"].fillna(False) & hits["inside_close"].fillna(False)
        else:
            hits["sweep_in_zone"] = hits["sweep"].fillna(False)
        st.session_state["hits"] = hits

if hits is not None and not hits.empty and "bear_above_%" not in hits.columns:
    frames = st.session_state.get("frames") or {}
    cache = {}
    for t in hits["ticker"].unique():
        if t not in frames:
            continue
        try:
            r = scan_ifvg(frames[t], include_ohlc=False, **_scan_kwargs(meta))
        except Exception:
            continue
        nr = r.get("nearest_red")
        cache[t] = {
            "bear_above_%": None if not nr else round(nr["pct"], 2),
            "bear_bot": None if not nr else nr["bot"],
            "bear_top": None if not nr else nr["top"],
        }
    if cache:
        extra = pd.DataFrame.from_dict(cache, orient="index")
        extra.index.name = "ticker"
        extra = extra.reset_index()
        hits = hits.merge(extra, on="ticker", how="left")
        st.session_state["hits"] = hits

if hits is not None and not hits.empty and "rsi" not in hits.columns:
    frames = st.session_state.get("frames") or {}
    cache = {}
    for t in hits["ticker"].unique():
        if t not in frames:
            continue
        try:
            r = scan_ifvg(frames[t], include_ohlc=False, **_scan_kwargs(meta))
        except Exception:
            continue
        rsi = r.get("rsi")
        dist = r.get("rsi_dist_30")
        cache[t] = {
            "rsi": None if rsi is None else round(float(rsi), 1),
            "rsi_dist_30": None if dist is None else round(float(dist), 1),
        }
    if cache:
        extra = pd.DataFrame.from_dict(cache, orient="index")
        extra.index.name = "ticker"
        extra = extra.reset_index()
        hits = hits.merge(extra, on="ticker", how="left")
        st.session_state["hits"] = hits

if hits is not None and not hits.empty and "eps_beat" not in hits.columns:
    hits = _attach_eps(hits, fetch_missing=bool(require_eps_beat))
    st.session_state["hits"] = hits

if hits is not None and not hits.empty and "adv_m" not in hits.columns:
    frames = st.session_state.get("frames") or {}
    as_of = (meta or {}).get("as_of")
    hits = attach_liquidity(hits, frames, as_of=as_of, fetch_mktcap=True)
    st.session_state["hits"] = hits

if hits is None:
    st.info("Set the ATR filter if you want, then hit **Scan**.")
    st.stop()

if hits.empty:
    as_of_txt = (meta or {}).get("as_of") or (hits["date"].iloc[0] if hits is not None and not hits.empty else "as-of bar")
    lb = int((meta or {}).get("lookback", 0))
    st.warning(
        f"No names overlapping an active green iFVG on {as_of_txt}"
        f" (lookback {lb})."
    )
    st.stop()

view = hits.copy()
if only_signal:
    view = view[view["lux_signal"]]
if only_sweep:
    col = "sweep_in_zone" if "sweep_in_zone" in view.columns else "sweep"
    view = view[view[col].fillna(False)]
view = view[view["dist_zone_%"] <= float(max_gap)]
if float(min_bear) > 0 and "bear_above_%" in view.columns and not view.empty:
    bear = view["bear_above_%"]
    view = view[bear.isna() | (bear >= float(min_bear))]
if int(max_rsi_dist) < 70:
    if "rsi_dist_30" not in view.columns:
        view = view.iloc[0:0]
    else:
        dist = view["rsi_dist_30"]
        view = view[dist.isna() | (dist <= float(max_rsi_dist))]
if require_eps_beat:
    if "eps_beat" not in view.columns:
        view = view.iloc[0:0]
    else:
        view = view[view["eps_beat"].fillna(False)]
if float(min_mktcap_bn) > 0:
    if "mktcap_bn" not in view.columns:
        view = view.iloc[0:0]
    else:
        # Missing cap = fail (don't sneak illiquid/unknown names through)
        view = view[view["mktcap_bn"].notna() & (view["mktcap_bn"] >= float(min_mktcap_bn))]
if float(min_adv_m) > 0:
    if "adv_m" not in view.columns:
        view = view.iloc[0:0]
    else:
        view = view[view["adv_m"].notna() & (view["adv_m"] >= float(min_adv_m))]
if nearest_only and not view.empty:
    view = (
        view.sort_values(["ticker", "dist_zone_%", "age_inv"])
        .groupby("ticker", as_index=False)
        .first()
    )
if not view.empty:
    sort_cols = ["inside_close", "dist_zone_%", "ticker"]
    sort_asc = [False, True, True]
    if "sweep_in_zone" in view.columns:
        sort_cols = ["sweep_in_zone", "inside_close", "dist_zone_%", "ticker"]
        sort_asc = [False, False, True, True]
    if show_lux and "lux_signal" in view.columns:
        sort_cols = ["lux_signal"] + sort_cols
        sort_asc = [False] + sort_asc
    view = view.sort_values(sort_cols, ascending=sort_asc)

if meta:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe", f"{meta['n_uni']} tickers")
    c2.metric("Priced (≥210d)", meta["n_priced"])
    c3.metric("Touching green iFVG", 0 if view.empty else int(view["ticker"].nunique()))
    if show_lux:
        lux_n = 0 if view.empty else int(view.loc[view["lux_signal"], "ticker"].nunique())
        c4.metric("LuxAlgo ▲ today", lux_n)
    else:
        inside_n = 0 if view.empty else int(view.loc[view["inside_close"], "ticker"].nunique())
        c4.metric("Close inside zone", inside_n)
    st.caption(f"Scan time {meta['elapsed']:.1f}s")
    uni = meta["uni"]
    st.caption(
        f"Constituents fetched {uni['fetched'][:10]}  ·  "
        f"S&P {uni['n_sp500']}  ·  NDX {uni['n_ndx100']}  ·  union {uni['n_union']}  ·  "
        f"as-of {hits['date'].iloc[0]}  ·  lookback {int(meta.get('lookback', 0))}"
    )
    if require_eps_beat and "eps_date" in hits.columns:
        n_src = int(hits.drop_duplicates("ticker")["eps_date"].notna().sum())
        if n_src == 0:
            st.warning(
                "EPS surprise cache has no last print for these hits, so the beat filter skips all of them. "
                "Same as Flow B: missing estimate = skip."
            )

if view.empty:
    st.warning(
        "Hits exist, but none pass the current filters. "
        "Loosen max % gap, liquidity mins, or uncheck the boxes."
    )
    st.stop()

st.subheader(f"{view['ticker'].nunique()} names  ·  {len(view)} zone hits")
if show_lux:
    lux_names = sorted(view.loc[view["lux_signal"], "ticker"].unique().tolist()) if "lux_signal" in view else []
    if lux_names:
        st.success("LuxAlgo bullish signal (close back above the green iFVG) on last bar: **" + ", ".join(lux_names) + "**")
if "sweep_in_zone" in view.columns:
    sweep_names = sorted(view.loc[view["sweep_in_zone"].fillna(False), "ticker"].unique().tolist())
    if sweep_names:
        st.success("Sweep + close inside green iFVG on last bar: **" + ", ".join(sweep_names) + "**")
st.caption(
    "Green iFVG = inverted **bearish** FVG that has not been body-traded through the bottom. "
    "Touch = as-of daily bar overlaps the zone. "
    "`bear_above_%` = % from close up to the nearest red iFVG (blank = none above). "
    "Sweep = last Low < prior 5-bar swing low and Close > that swing. "
    "Sweep filter also requires Close inside the green iFVG. "
    "`rsi_dist_30` = RSI(14) − 30 (negative = below 30). "
    "`eps_surprise` = last reported Yahoo EPS vs consensus (Flow B rule: > 0 to pass). "
    "`mktcap_bn` = Yahoo market cap ($B). "
    "`adv_m` = 20-session average Close×Volume ($M). Missing liquidity fields fail the filter when it is on. "
    "Midline is not a filter."
)
show = view.copy()
show.insert(0, "tag", show["ticker"].map(lambda t: TAG_SYM[_tag_of(t)]))
show["sp500"] = show["sp500"].map({True: "Y", False: ""})
show["ndx100"] = show["ndx100"].map({True: "Y", False: ""})
show["inside_close"] = show["inside_close"].map({True: "Y", False: ""})
show["wick_only"] = show["wick_only"].map({True: "Y", False: ""})
if show_lux and "lux_signal" in show.columns:
    show["lux_signal"] = show["lux_signal"].map({True: "Y", False: ""})
elif "lux_signal" in show.columns:
    show = show.drop(columns=["lux_signal"])
if "sweep" in show.columns:
    show["sweep"] = show["sweep"].map({True: "Y", False: ""})
if "sweep_in_zone" in show.columns:
    show["sweep_in_zone"] = show["sweep_in_zone"].map({True: "Y", False: ""})
if "eps_beat" in show.columns:
    show["eps_beat"] = show["eps_beat"].map({True: "Y", False: ""})
# Raw USD fields stay in CSV; table shows $B / $M only.
show = show.drop(columns=[c for c in ("market_cap", "adv_dollar") if c in show.columns])
edited = st.data_editor(
    show,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=[c for c in show.columns if c != "tag"],
    column_config={
        "tag": st.column_config.SelectboxColumn(
            "tag",
            options=[TAG_NONE, "🟢", "🟡", "🔴"],
            required=True,
            width="small",
            help="Click: none, green, yellow, or red.",
        ),
        "close": st.column_config.NumberColumn(format="%.2f"),
        "ifvg_top": st.column_config.NumberColumn(format="%.2f"),
        "ifvg_bot": st.column_config.NumberColumn(format="%.2f"),
        "ifvg_mid": st.column_config.NumberColumn(format="%.2f"),
        "bear_above_%": st.column_config.NumberColumn(format="%.2f"),
        "bear_bot": st.column_config.NumberColumn(format="%.2f"),
        "bear_top": st.column_config.NumberColumn(format="%.2f"),
        "swing_low": st.column_config.NumberColumn(format="%.2f"),
        "rsi": st.column_config.NumberColumn(format="%.1f"),
        "rsi_dist_30": st.column_config.NumberColumn(format="%.1f"),
        "eps_surprise": st.column_config.NumberColumn(
            "EPS surprise",
            format="+0.0%",
            help="Yahoo EPS surprise on the last reported print. Missing = blank.",
        ),
        "mktcap_bn": st.column_config.NumberColumn(
            "Mkt cap ($B)",
            format="%.1f",
            help="Yahoo market cap in billions.",
        ),
        "adv_m": st.column_config.NumberColumn(
            "ADV ($M)",
            format="%.1f",
            help="20-session average dollar volume (Close × Volume) in millions.",
        ),
    },
    key=f"hits_tag_editor_{hash(tuple(show['ticker'].astype(str)))}",
)
_sync_tags_from_editor(edited)
csv_df = view.copy()
csv_df.insert(0, "tag", csv_df["ticker"].map(lambda t: _tag_of(t)))
st.download_button(
    "CSV",
    csv_df.to_csv(index=False).encode("utf-8"),
    file_name="green_ifvg_hits.csv",
    mime="text/csv",
)

st.subheader("Chart")
c_pick, c_last = st.columns([3, 1])
with c_pick:
    pick = st.selectbox(
        "Ticker",
        sorted(view["ticker"].unique()),
        format_func=lambda t: f"{TAG_SYM[_tag_of(t)]} {t}" if _tag_of(t) else str(t),
    )
with c_last:
    show_last = st.number_input(
        "Show last (LuxAlgo)",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="Display cap only — last N green and last N red iFVGs, same as LuxAlgo Show Last. Scan still uses every active green.",
    )
frames = st.session_state["frames"]
if pick in frames:
    res = scan_ifvg(frames[pick], include_ohlc=True, **_scan_kwargs(meta))
    st.plotly_chart(
        _chart(res, pick, show_last=int(show_last), show_signals=show_lux),
        use_container_width=True,
    )
    st.caption(
        "Two-tone boxes: original FVG color from **formed → inverted**, "
        "flipped color from **inverted → now**. Red iFVGs included. "
        + ("▲/▼ are LuxAlgo retest signals. " if show_lux else "")
        + "Gold dotted line is the prior 5-bar swing low used by the sweep filter. "
        "Scan table is unchanged (all active greens, proximity filter)."
    )
    shown_g = _last_n(res.get("green_zones", []), int(show_last))
    shown_r = _last_n(res.get("red_zones", []), int(show_last))
    zdf = pd.DataFrame(shown_g + shown_r)
    if not zdf.empty:
        drop = [c for c in ("left_i", "inv_i", "signals") if c in zdf.columns]
        st.dataframe(zdf.drop(columns=drop), hide_index=True, use_container_width=True)
