# India Portfolio Lab — Multi-Strategy Allocation & Risk Engine

An interactive portfolio-construction and risk engine for Indian NSE indices, built on the
methodology of the EDHEC *Investment Management with Python and Machine Learning* specialization.

### ▶ Live app: **https://paterson-risk-engine.streamlit.app**

No installation needed — it opens in the browser.

---

## What it does

An 8-tab interactive dashboard that applies the full EDHEC toolkit to Indian markets, with a
plain-English footnote on every tab explaining the method, its pros and cons, and how it maps to
real-world practice.

- **Universe** — 11 NSE sector indices (Auto, Bank, FMCG, IT, Media, Metal, Pharma, PSU Bank, Realty,
  Energy, Infra) plus Nifty50/100/Next50/500 and Midcap50/100 benchmarks; monthly, 2011–2026.
- **7 allocation strategies** — Equal Weight, Global Minimum Variance, Max Sharpe (tangency),
  Max Diversification (Choueifaty MDP), Risk Parity (ERC), Hierarchical Risk Parity (Lopez de Prado
  ML clustering), and Black-Litterman equilibrium.
- **4 covariance estimators** — sample, Ledoit-Wolf shrinkage, constant-correlation (Elton-Gruber),
  and EWMA (RiskMetrics) volatility forecasting.
- **Efficient frontier + Capital Market Line**, every strategy plotted at its risk/return point.
- **Walk-forward, out-of-sample backtest** with transaction costs on turnover, cost-drag reporting,
  and rolling vs. expanding windows. Ranked Sharpe / Sortino / Calmar / Max-Drawdown table.
- **Regime analysis** — a Gaussian-mixture model separates calm/bull from turbulent/bear months.
- **CPPI / TIPP insurance overlay** — dynamic floor-protected allocation with an optional ratcheting floor.
- **ALM / Pension tab** — funding ratio vs. interest rates and duration-matching (LDI) immunization.

## The eight tabs

| Tab | What it does |
|---|---|
| Overview | Per-asset return, vol, Sharpe/Sortino/Calmar, VaR & CVaR, and the correlation heatmap. |
| Efficient Frontier + CML | The Markowitz frontier and Capital Market Line, with every strategy marked. |
| Strategy Lab | Pick one strategy; see its weights, concentration, and an idea/pros/cons/real-world card. |
| Backtest | Out-of-sample test of all strategies with costs, turnover and drawdowns; wealth curves. |
| Regime Analysis | Gaussian-mixture split into calm vs turbulent regimes, with per-regime stats. |
| CPPI Insurance | Downside-protection overlay with dynamic de-risking and an optional ratcheting floor. |
| ALM / Pension | Funding-ratio-vs-rates curve and duration matching for a liability stream (LDI). |
| Notes | Full methodology, covariance-estimator notes, and data-source details. |

## Methodology, mapped to the EDHEC courses

- **Course 1** — return/risk analytics (Sharpe/Sortino/Calmar, VaR/CVaR, drawdown), the mean-variance
  efficient frontier and CML, CPPI/TIPP insurance, and ALM (funding ratio, duration matching).
- **Course 2** — robust covariance (Ledoit-Wolf shrinkage, constant-correlation, EWMA) and
  Black-Litterman equilibrium returns (market-implied, not noisy historical means).
- **Course 3** — Hierarchical Risk Parity (clustering-based ML allocation) and regime analysis
  (Gaussian mixture).

Weight optimization uses `scipy.optimize.minimize` (SLSQP) for GMV/MSR/ERC/MDP and hierarchical
clustering + recursive bisection for HRP. Every backtest is strictly walk-forward: weights at month
*t* use only data before *t*, then apply to *t*'s realised return; weights drift between rebalances
and transaction costs are charged only on the actual trade back to target, so the return path and
cost path are internally consistent.

**Headline finding:** across Indian sector data, the strategies that do *not* forecast expected
returns (HRP, Risk Parity, GMV, Equal Weight) consistently beat Max Sharpe out-of-sample — a clean
demonstration that historical mean returns are too noisy to optimize on directly, which is the whole
reason robust methods exist.

## Data

NSE sector and benchmark index returns are sourced from official NSE index data (monthly, 2011–2026)
and are **price-return (ex-dividend)** series. They are validated against real-world published Nifty 50
figures — the March-2020 COVID crash reads −23.25% and calendar-year returns track the published price
index year-for-year (2020 +14.9%, 2021 +24.1%, 2022 +4.3%). A total-return (TRI) basis would add
roughly 1.2–1.5%/year (the dividend yield), so the figures here are, if anything, modestly conservative.
The risk-free rate is a user-set input (India 10Y G-Sec proxy), held flat rather than time-varying.

## Repository

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (the deployable app) |
| `portfolio_engine.py` | Strategy, risk, and backtest engine |
| `ind_in_m_sectors.csv` | Indian sector monthly returns |
| `ind_in_m_benchmarks.csv` | Indian benchmark index monthly returns |
| `requirements.txt` | Python dependencies |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```
