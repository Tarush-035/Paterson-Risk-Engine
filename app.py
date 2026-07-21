"""
Indian Multi-Strategy Portfolio & Risk Engine
-----------------------------------------------
Streamlit dashboard for the EDHEC "Investment Management with Python and ML"
project (Paterson Securities internship).

Run locally:    streamlit run app.py
Deploy:         push this repo to GitHub, then deploy at share.streamlit.io
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import portfolio_engine as pe

st.set_page_config(page_title="Indian Portfolio & Risk Engine", layout="wide")

# ============================================================
# DATA LOADING (cached)
# ============================================================

@st.cache_data
def load_all():
    sectors = pe.load_returns("ind_in_m_sectors.csv")
    benchmarks = pe.load_returns("ind_in_m_benchmarks.csv")
    return sectors, benchmarks


sectors, benchmarks = load_all()

# ============================================================
# SIDEBAR — CONTROLS
# ============================================================

st.sidebar.title("Portfolio & Risk Engine")
st.sidebar.caption("Indian sector indices · EDHEC risk-kit methodology")

universe_choice = st.sidebar.radio("Universe", ["Sectors", "Benchmarks"], index=0)
universe = sectors if universe_choice == "Sectors" else benchmarks

all_assets = universe.columns.tolist()
default_assets = all_assets if len(all_assets) <= 8 else all_assets[:8]
selected_assets = st.sidebar.multiselect("Assets", all_assets, default=default_assets)

if len(selected_assets) < 2:
    st.warning("Select at least 2 assets from the sidebar to continue.")
    st.stop()

data = universe[selected_assets].dropna()

st.sidebar.markdown("---")
strategy_name = st.sidebar.selectbox("Strategy", list(pe.STRATEGIES.keys()), index=4)
cov_method = st.sidebar.selectbox(
    "Covariance estimator", ["sample", "shrinkage", "ewma"], index=1,
    format_func=lambda x: {
        "sample": "Sample (historical)",
        "shrinkage": "Ledoit-Wolf shrinkage",
        "ewma": "EWMA (vol-forecast, λ=0.94)",
    }[x],
)
riskfree_rate = st.sidebar.slider("Risk-free rate (India 10Y G-Sec proxy)", 0.02, 0.10, 0.065, 0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest settings")
window = st.sidebar.slider("Estimation window (months)", 24, 84, 36, 6)
rebalance_every = st.sidebar.slider("Rebalance every (months)", 1, 12, 3, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("CPPI insurance overlay")
use_cppi = st.sidebar.checkbox("Apply CPPI to backtested strategy", value=False)
cppi_m = st.sidebar.slider("Multiplier (m)", 1.0, 6.0, 3.0, 0.5) if use_cppi else 3.0
cppi_floor = st.sidebar.slider("Floor (% of start capital)", 0.50, 0.95, 0.80, 0.05) if use_cppi else 0.80

st.title(f"{universe_choice}: Indian Multi-Strategy Portfolio & Risk Engine")

# ============================================================
# TABS
# ============================================================

tab_overview, tab_ef, tab_backtest, tab_compare, tab_cppi = st.tabs(
    ["Overview", "Efficient Frontier", "Backtest", "Strategy Comparison", "CPPI"]
)

# ---------- OVERVIEW ----------
with tab_overview:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Correlation matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data.corr(), cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(selected_assets)), selected_assets, rotation=45, ha="right")
        ax.set_yticks(range(len(selected_assets)), selected_assets)
        for i in range(len(selected_assets)):
            for j in range(len(selected_assets)):
                ax.text(j, i, f"{data.corr().iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        st.pyplot(fig)
    with col2:
        st.subheader("Per-asset summary")
        st.dataframe(pe.summary_stats(data, riskfree_rate=riskfree_rate).round(4), use_container_width=True)

    st.caption(f"Data: {data.shape[0]} months, {data.index[0]} to {data.index[-1]}")

# ---------- EFFICIENT FRONTIER ----------
with tab_ef:
    er = pe.annualize_rets(data, 12)
    cov = pe.COV_METHODS[cov_method](data)

    n_points = 40
    target_rs = np.linspace(er.min(), er.max(), n_points)
    vols = []
    for tr in target_rs:
        w = pe._minimize_vol(tr, er.values, cov.values)
        vols.append(pe.portfolio_vol(w, cov.values))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(vols, target_rs, "b.-", label="Efficient Frontier")

    markers = {
        "EW": (pe.weights_ew(data), "gold", "o"),
        "GMV": (pe.weights_gmv(data, cov_method=cov_method), "navy", "s"),
        "MSR": (pe.weights_msr(data, riskfree_rate=riskfree_rate, cov_method=cov_method), "green", "^"),
        "ERC": (pe.weights_risk_parity(data, cov_method=cov_method), "purple", "D"),
        "HRP": (pe.weights_hrp(data, cov_method=cov_method), "crimson", "*"),
    }
    for label, (w, color, marker) in markers.items():
        r = pe.portfolio_return(w, er.values)
        v = pe.portfolio_vol(w, cov.values)
        ax.scatter([v], [r], color=color, marker=marker, s=120, label=label, zorder=5)

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title(f"Efficient Frontier — {cov_method} covariance")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Portfolio weights by strategy")
    weight_table = pd.DataFrame({label: w for label, (w, _, _) in markers.items()}, index=selected_assets)
    st.dataframe((weight_table * 100).round(2), use_container_width=True)

# ---------- BACKTEST ----------
with tab_backtest:
    strategy_fn = pe.STRATEGIES[strategy_name]
    try:
        wealth, weight_hist = pe.backtest_strategy(
            data, strategy_fn, window=window, rebalance_every=rebalance_every,
            cov_method=cov_method, riskfree_rate=riskfree_rate,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    bench_wealth, _ = pe.backtest_strategy(data, pe.weights_ew, window=window, rebalance_every=rebalance_every)

    strat_rets = pe.wealth_to_returns(wealth)
    stats = pe.summary_stats(strat_rets, riskfree_rate=riskfree_rate)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Return", f"{stats['Annualized Return']*100:.2f}%")
    col2.metric("Annualized Vol", f"{stats['Annualized Vol']*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
    col4.metric("Max Drawdown", f"{stats['Max Drawdown']*100:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(wealth.index.to_timestamp(), wealth.values, label=strategy_name, linewidth=2)
    ax.plot(bench_wealth.index.to_timestamp(), bench_wealth.values, label="Equal Weight (benchmark)",
            linewidth=1.5, linestyle="--", color="gray")
    ax.set_ylabel("Wealth (₹, start=1000)")
    ax.set_title(f"Backtested wealth — {strategy_name} ({cov_method} cov, rebalanced every {rebalance_every}m)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    dd = pe.drawdown(strat_rets)
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.fill_between(dd.index.to_timestamp(), dd["Drawdown"].values * 100, 0, color="crimson", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    st.subheader("Weight evolution")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.stackplot(weight_hist.index.to_timestamp(), weight_hist.T.values, labels=weight_hist.columns)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax3.set_ylabel("Weight")
    ax3.set_ylim(0, 1)
    st.pyplot(fig3)

    st.subheader("Full stats")
    st.dataframe(stats.to_frame("Value").round(4), use_container_width=True)

# ---------- STRATEGY COMPARISON ----------
with tab_compare:
    st.subheader(f"All strategies — {window}m window, rebalanced every {rebalance_every}m, {cov_method} covariance")
    rows = []
    wealth_curves = {}
    for name, fn in pe.STRATEGIES.items():
        w, _ = pe.backtest_strategy(data, fn, window=window, rebalance_every=rebalance_every,
                                     cov_method=cov_method, riskfree_rate=riskfree_rate)
        r = pe.wealth_to_returns(w)
        s = pe.summary_stats(r, riskfree_rate=riskfree_rate)
        rows.append({"Strategy": name, **s.to_dict()})
        wealth_curves[name] = w

    comp_df = pd.DataFrame(rows).set_index("Strategy")
    st.dataframe(comp_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, w in wealth_curves.items():
        ax.plot(w.index.to_timestamp(), w.values, label=name, linewidth=1.8)
    ax.set_ylabel("Wealth (₹, start=1000)")
    ax.set_title("Cumulative wealth — all strategies")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---------- CPPI ----------
with tab_cppi:
    st.subheader("CPPI insurance overlay")
    st.caption(
        "Applies a Constant Proportion Portfolio Insurance overlay on top of the "
        "backtested strategy's return stream — dynamically shifts between the risky "
        "strategy and a risk-free safe asset to protect a floor value."
    )
    strategy_fn = pe.STRATEGIES[strategy_name]
    wealth, _ = pe.backtest_strategy(data, strategy_fn, window=window, rebalance_every=rebalance_every,
                                      cov_method=cov_method, riskfree_rate=riskfree_rate)
    risky_r = pe.wealth_to_returns(wealth).to_frame(strategy_name)

    cppi = pe.run_cppi(risky_r, m=cppi_m, start=1000, floor=cppi_floor, riskfree_rate=riskfree_rate)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(cppi["Wealth"].index.to_timestamp(), cppi["Wealth"][strategy_name], label="CPPI Wealth", linewidth=2)
    ax.plot(cppi["Risky Wealth"].index.to_timestamp(), cppi["Risky Wealth"][strategy_name],
            label=f"{strategy_name} (no insurance)", linewidth=1.5, linestyle="--")
    ax.plot(cppi["Floor Value"].index.to_timestamp(), cppi["Floor Value"][strategy_name],
            label="Floor", linewidth=1.2, linestyle=":", color="red")
    ax.set_ylabel("Wealth (₹, start=1000)")
    ax.set_title(f"CPPI overlay (m={cppi_m}, floor={cppi_floor*100:.0f}%)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(cppi["Risky Allocation"].index.to_timestamp(), cppi["Risky Allocation"][strategy_name] * 100,
             color="green", linewidth=2)
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Risky allocation (%)")
    ax2.set_title("CPPI risky allocation over time")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    final_cppi = cppi["Wealth"][strategy_name].iloc[-1]
    final_risky = cppi["Risky Wealth"][strategy_name].iloc[-1]
    col1, col2 = st.columns(2)
    col1.metric("CPPI final wealth", f"₹{final_cppi:,.0f}")
    col2.metric(f"{strategy_name} final wealth (no insurance)", f"₹{final_risky:,.0f}")

st.markdown("---")
st.caption(
    "Methodology extends the EDHEC 'Introduction to Portfolio Construction and Analysis with Python' "
    "risk kit: mean-variance optimization (GMV/MSR), Equal Risk Contribution, Hierarchical Risk Parity "
    "(Lopez de Prado), Ledoit-Wolf shrinkage and EWMA covariance forecasting, and CPPI insurance strategies, "
    "applied to Indian NSE sector indices."
)
