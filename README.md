# India Portfolio Lab — Multi-Strategy Allocation & Risk Engine

An interactive portfolio-construction and risk engine for Indian markets and a multi-asset pool,
built on the methodology of the EDHEC *Investment Management with Python and Machine Learning*
specialization.

### ▶ Live app: **https://paterson-risk-engine.streamlit.app**

No installation needed — it opens in the browser.

---

## What it does

An 8-tab interactive dashboard that applies the full EDHEC toolkit across three selectable universes,
with a plain-English footnote on every tab explaining the method, its pros and cons, and how it maps
to real-world practice.

- **Three universes** (switchable in the sidebar):
  - **Sectors** — 11 NSE sector indices (Auto, Bank, FMCG, IT, Media, Metal, Pharma, PSU Bank, Realty, Energy, Infra).
  - **Benchmarks** — Nifty 50/100/Next 50/500 and Midcap 50/100 (heavily overlapping).
  - **Multi-Asset** — a 5-asset cross-asset pool (India equity, US equity in INR, gold, 10Y G-Sec, cash) — the only universe with genuinely uncorrelated building blocks.
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
| Notes | Full methodology, covariance-estimator notes, proxies, and data-source details. |

## The Multi-Asset pool — assets, proxies & conversions

The reason equity-only universes underperform is that Indian sectors are all highly correlated
(~0.6–0.85) and crash together — diversification abandons you exactly when you need it. The
Multi-Asset universe fixes this with genuinely uncorrelated asset classes:

| Asset | Instrument / proxy | Return type | Conversion |
|---|---|---|---|
| India equity | Nifty 50 index | Price (ex-dividend) | native INR |
| US equity | S&P 500 (`^GSPC`) × USD-INR | Price (ex-dividend) | `(1+r_USD)(1+Δfx)−1` |
| Gold | GOLDBEES ETF (Nippon, NSE) | Total | native INR |
| 10Y G-Sec | ICICI Prudential Gilt Fund NAV | Total (net of fees) | monthly NAV change |
| Cash | HDFC Liquid Fund NAV | Total (net of fees) | monthly NAV change |

The USD→INR conversion is multiplicative and monthly (unit-tested: `(1−0.05)(1+0.04)−1 = −1.20%`).
Defensive-sleeve data glitches (a monthly move beyond ±40% on bonds/cash/gold) are auto-cleaned by
interpolation. All series are aligned to a common monthly window (Oct 2011 – Apr 2026).

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

## Headline findings

- **Within equity (Sectors / Benchmarks):** the strategies that do *not* forecast expected returns
  (HRP, Risk Parity, GMV, Equal Weight) consistently beat Max Sharpe out-of-sample — historical mean
  returns are too noisy to optimize on directly, which is the whole reason robust methods exist. But
  none beats a simple buy-and-hold on *raw return* — the value is risk control, not alpha.
- **Multi-Asset is where it pays off:** adding uncorrelated assets roughly **tripled the Sharpe ratio**
  (e.g. Equal Weight from ~0.2 to ~0.8) and **cut max drawdown from ~40% to single digits**, at
  similar returns — a clean demonstration that genuine cross-asset diversification, not equity
  reweighting, is the real source of risk-adjusted improvement.

## Data & caveats

- Indian indices: official NSE data, **price-return (ex-dividend)**; validated against published Nifty 50
  figures (Mar-2020 = −23.25%; 2020 +14.9%, 2021 +24.1%, 2022 +4.3%).
- In the Multi-Asset pool the two equities are price-return while bonds/gold/cash are total-return
  (their natural form) — a documented, minor inconsistency that doesn't affect the correlations or the risk story.
- The **risk-free rate** is a flat slider input — the *scoring hurdle* for Sharpe/Sortino and an input
  to Max Sharpe / Black-Litterman / CPPI. It is **not** a time-varying series and **not** an asset in the pool
  (the G-Sec and cash sleeves are the tradable fixed-income assets; the rf is a separate benchmark).
- Costs are flat basis points on turnover; **taxes (STT, STCG/LTCG) are not modelled** — which favours buy-and-hold.
- Regime labels and efficient-frontier markers are in-sample/illustrative; the honest performance numbers are on the Backtest tab.

## Repository

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (the deployable app) |
| `portfolio_engine.py` | Strategy, risk, and backtest engine |
| `ind_in_m_sectors.csv` | Indian sector monthly returns |
| `ind_in_m_benchmarks.csv` | Indian benchmark index monthly returns |
| `ind_in_m_pool.csv` | Multi-asset pool monthly returns (India eq, US eq INR, gold, G-Sec, cash) |
| `requirements.txt` | Python dependencies |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```
