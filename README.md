# Indian Multi-Strategy Portfolio & Risk Engine

Built for the EDHEC "Investment Management with Python and Machine Learning" course project
(Paterson Securities internship). Extends the course's `edhec_risk_kit.py` methodology and
applies it to Indian NSE sector indices.

## What it does

- **Universe**: 11 Indian NSE sector indices (Auto, Bank, FMCG, IT, Media, Metal, Pharma,
  PSU Bank, Realty, Energy, Infra), monthly returns 2011–2026, plus Nifty50/100/Next50/500,
  Midcap50/100 benchmarks.
- **Strategies**: Equal Weight, Global Minimum Variance, Max Sharpe (tangency), Equal Risk
  Contribution / Risk Parity, and Hierarchical Risk Parity (Lopez de Prado clustering method).
- **Covariance estimation**: sample (historical), Ledoit-Wolf shrinkage, and EWMA
  (RiskMetrics-style) volatility forecasting — the EWMA option is the "predictive" layer:
  it reacts to changing volatility regimes (e.g. COVID) instead of weighting all history equally.
- **Backtesting**: rolling walk-forward backtest with configurable estimation window and
  rebalance frequency, benchmarked against equal-weight.
- **CPPI insurance overlay**: dynamic floor-protected allocation between a chosen strategy
  and a risk-free asset.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (the deployable app) |
| `portfolio_engine.py` | Strategy, risk, and backtest engine (no notebook-only dependencies) |
| `ind_in_m_sectors.csv` | Indian sector monthly returns |
| `ind_in_m_benchmarks.csv` | Indian benchmark index monthly returns |
| `requirements.txt` | Python dependencies for deployment |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy as a public website (no coding/dev tools needed)

**1. Put the 5 files above into a GitHub repository.**
   - Go to github.com, sign in (create a free account if you don't have one).
   - Click the **+** in the top right → **New repository**. Name it (e.g. `india-portfolio-engine`),
     set it to **Public**, click **Create repository**.
   - On the new repo page, click **Add file → Upload files**.
   - Drag in all 5 files (`app.py`, `portfolio_engine.py`, `ind_in_m_sectors.csv`,
     `ind_in_m_benchmarks.csv`, `requirements.txt`). Click **Commit changes**.
   - No git command line needed — this is all done through the browser.

**2. Deploy on Streamlit Community Cloud (free).**
   - Go to **share.streamlit.io**, sign in with your GitHub account.
   - Click **New app**, pick your repo, branch `main`, and set the main file to `app.py`.
   - Click **Deploy**. It builds for ~1–2 minutes and gives you a public URL
     (e.g. `https://your-app-name.streamlit.app`) you can share with your instructor.

**3. Updating later**: edit/upload files in the GitHub repo (again via the web "Upload files"
   button, or "Edit" pencil icon on a file) — Streamlit Cloud auto-redeploys on every commit.

## Methodology notes for submission

- Weight optimization uses `scipy.optimize.minimize` (SLSQP) for GMV/MSR/Risk Parity, and
  hierarchical clustering (`scipy.cluster.hierarchy`) + recursive bisection for HRP.
- Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`) addresses estimation error in the
  sample covariance matrix, which matters here given the relatively short (~13-year) history.
- EWMA covariance forecast: `Σ_t = λ·Σ_(t-1) + (1-λ)·r_(t-1)r_(t-1)'`, λ=0.94 (RiskMetrics
  default) — a lightweight, explainable "predictive" volatility model, in contrast to a
  black-box return-prediction model.
- CPPI mechanics follow the EDHEC course's constant-proportion insurance formula:
  risky allocation = m × (account value − floor value), clipped to [0, 1] of the portfolio.

## What still needs verification before submission

- NSE sector index tickers pulled via `yfinance` — worth spot-checking a few months against
  official NSE index factsheets, since Yahoo's index data quality/adjustments aren't guaranteed.
- Backtest results are gross of transaction costs (a `transaction_cost` parameter exists in
  `backtest_strategy()` but defaults to 0) — decide whether to report costed or uncosted numbers.
- Risk-free rate is a fixed slider input (India 10Y G-Sec proxy), not a time-varying series —
  fine for a course project, but flag it as a simplification if asked.
