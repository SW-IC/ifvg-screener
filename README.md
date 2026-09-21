---
title: iFVG Screener
emoji: 🟢
colorFrom: green
colorTo: red
sdk: docker
app_port: 8501
pinned: false
---

# iFVG screener

Streamlit daily screener for inversion FVGs (LuxAlgo IFVG). **Bull** = green iFVG (support) + optional EPS surprise > 0. **Bear** = red iFVG (resistance) + optional EPS surprise < 0. Universe = S&P 500 ∪ Nasdaq-100. Hits Yahoo for prices / EPS on first load (price parquet cache is local-only and not shipped).

## Live price (unfinished candle)

**Live price (today's unfinished candle)** is on by default. While the US session is open:

- the daily history comes from the cached pull, and today's row is replaced by a fresh Yahoo quote (open / high / low / last / volume so far);
- the scan runs on that in-progress candle, so a name can appear or drop out mid-session;
- the open page re-fetches and re-scans every *Refresh every (minutes)* (default 5). The caption under the metrics shows when prices were fetched, in US Eastern time.

**Lookback (sessions)** defaults to `0` = the latest bar, which is today's unfinished candle in live mode. Set it to `1` to go back to the last completed session.

Turn the checkbox off to go back to plain daily bars from the cached download (up to 6 hours old). *Refresh prices now* forces an immediate live pull.

## Local

```powershell
cd C:\Users\ian09\grok\ifvg-screener
python -m pip install -r requirements.txt
streamlit run app.py
```

Public URL while this PC is on:

```powershell
.\run-tunnel.ps1
```

## Streamlit Community Cloud (free shareable link)

1. Push this folder to GitHub:

```powershell
.\deploy.ps1 -GitHub
```

2. Open https://share.streamlit.io → **New app**.
3. Repo = this repo, branch = default, **Main file path** = `app.py`.
4. Deploy. URL looks like `https://<name>.streamlit.app`.

First Scan downloads daily bars (can take a few minutes). Earnings surprise uses `cache/earnings_surprise.pkl` when present.

## Hugging Face Spaces

Docker Spaces need **HF PRO**. Prefer Streamlit Cloud above.

```powershell
python -c "from huggingface_hub import login; login()"
.\deploy.ps1 -HuggingFace
```

Space URL: `https://huggingface.co/spaces/<you>/ifvg-screener`
