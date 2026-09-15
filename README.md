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
