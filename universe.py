"""S&P 500 ∪ Nasdaq-100 constituents (Yahoo symbols)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "ifvg-screener/0.1 (prototype; research)"}


def _yahoo(sym: str) -> str:
    return str(sym).strip().replace(".", "-").upper()


def _read_html(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def _sp500() -> list[str]:
    tables = _read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        for key in ("symbol", "ticker"):
            if key in cols:
                s = t[t.columns[cols.index(key)]].astype(str)
                return [_yahoo(x) for x in s if x and x.lower() != "nan"]
    raise RuntimeError("S&P 500 table not found")


def _ndx() -> list[str]:
    tables = _read_html("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies")
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        for key in ("ticker", "symbol"):
            if key in cols:
                s = t[t.columns[cols.index(key)]].astype(str)
                out = [_yahoo(x) for x in s if x and x.lower() != "nan" and len(x) <= 6]
                if len(out) >= 80:
                    return out
    raise RuntimeError("Nasdaq-100 table not found")


def load_universe(force: bool = False) -> dict:
    path = CACHE / "universe.json"
    if path.exists() and not force:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(data["fetched"])
        if datetime.now(timezone.utc) - ts < timedelta(days=7):
            return data

    sp = sorted(set(_sp500()))
    ndx = sorted(set(_ndx()))
    both = sorted(set(sp) | set(ndx))
    data = {
        "fetched": datetime.now(timezone.utc).isoformat(),
        "sp500": sp,
        "ndx100": ndx,
        "union": both,
        "n_sp500": len(sp),
        "n_ndx100": len(ndx),
        "n_union": len(both),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
