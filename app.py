"""
Indian Multi-Strategy Portfolio & Risk Engine
===============================================
Interactive dashboard implementing the EDHEC "Investment Management with Python
and Machine Learning" toolkit on Indian NSE sector indices.

Covers, practically and side-by-side:
  - Course 1: returns/risk analytics, mean-variance frontier + CML, CPPI insurance, ALM/LDI
  - Course 2: robust covariance (shrinkage, constant-correlation, EWMA), Black-Litterman
  - Course 3: HRP (ML clustering), regime analysis (Gaussian mixture)

Run locally:  streamlit run app.py
Deploy:       push repo to GitHub -> share.streamlit.io
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import portfolio_engine as pe

st.set_page_config(page_title="Indian Portfolio & Risk Engine", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

@st.cache_data
def load_all():
    sectors = pe.load_returns("ind_in_m_sectors.csv")
    benchmarks = pe.load_returns("ind_in_m_benchmarks.csv")
    return sectors, benchmarks


def footnote(text):
    st.caption(text)


def strategy_card(name):
    info = pe.STRATEGY_INFO.get(name)
    if not info:
        return
    st.markdown(f"**{name}**")
    st.markdown(f"*Idea:* {info['idea']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Pros**  \n{info['pros']}")
    with c2:
        st.markdown(f"**Cons**  \n{info['cons']}")
    st.markdown(f"*In the real world:* {info['real_world']}")


sectors, benchmarks = load_all()

# ------------------------------------------------------------
# Sidebar — global controls
# ------------------------------------------------------------

st.sidebar.title("Controls")
st.sidebar.caption("Indian NSE indices · EDHEC methodology")

universe_choice = st.sidebar.radio("Universe", ["Sectors", "Benchmarks"], index=0,
                                    help="Sectors = 11 NSE sector indices. Benchmarks = Nifty50/100/500 etc.")
universe = sectors if universe_choice == "Sectors" else benchmarks

all_assets = universe.columns.tolist()
default_assets = all_assets if len(all_assets) <= 8 else all_assets[:8]
selected_assets = st.sidebar.multiselect("Assets in the portfolio", all_assets, default=default_assets)

if len(selected_assets) < 2:
    st.warning("Select at least 2 assets in the sidebar to continue.")
    st.stop()

data = universe[selected_assets].dropna()

st.sidebar.markdown("---")
cov_method = st.sidebar.selectbox(
    "Covariance estimator", list(pe.COV_METHODS.keys()), index=1,
    format_func=lambda x: {
        "sample": "Sample (historical)",
        "shrinkage": "Ledoit-Wolf shrinkage",
        "constant_corr": "Constant correlation",
        "ewma": "EWMA (vol forecast)",
    }[x],
    help="How the risk (covariance) matrix is estimated. Drives every optimizer.",
)
footnote_cov = pe.COV_INFO[cov_method]
st.sidebar.caption(footnote_cov)

riskfree_rate = st.sidebar.slider("Risk-free rate (India 10Y G-Sec proxy)", 0.00, 0.10, 0.065, 0.005,
                                  help="~6.5-6.8% currently. Sharpe ratios are very sensitive to this.")

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest settings")
window = st.sidebar.slider("Estimation window (months)", 24, 84, 36, 6)
window_type = st.sidebar.radio("Window type", ["rolling", "expanding"], index=0,
                               help="Rolling = fixed lookback. Expanding = all history to date.")
rebalance_every = st.sidebar.slider("Rebalance every (months)", 1, 12, 3, 1)
tc_bps = st.sidebar.slider("Transaction cost (bps, one-way)", 0, 50, 10, 5,
                           help="India cash-equity all-in costs ~10-30 bps. Applied to turnover.")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("Indian Multi-Strategy Portfolio & Risk Engine")
st.caption(
    f"Universe: **{universe_choice}** · {data.shape[1]} assets · "
    f"{data.shape[0]} months ({data.index[0]} to {data.index[-1]}) · "
    f"covariance: **{cov_method}** · Rf: **{riskfree_rate*100:.1f}%**"
)

tabs = st.tabs([
    "Overview", "Efficient Frontier + CML", "Strategy Lab",
    "Backtest", "Regime Analysis", "CPPI Insurance", "ALM / Pension", "Notes",
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tabs[0]:
    st.subheader("The assets, before any optimization")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Per-asset annualized stats**")
        st.dataframe(pe.summary_stats(data, riskfree_rate=riskfree_rate).round(3),
                     use_container_width=True)
        footnote(
            "Sharpe = (annualized return − risk-free) / annualized vol. "
            "Sortino uses only downside vol; Calmar uses max drawdown. "
            "Negative-skew, high-kurtosis assets (crash-prone) look worse on VaR/CVaR."
        )
    with c2:
        st.markdown("**Correlation matrix**")
        fig, ax = plt.subplots(figsize=(6, 5))
        corr = data.corr()
        im = ax.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(selected_assets)), selected_assets, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(selected_assets)), selected_assets, fontsize=8)
        for i in range(len(selected_assets)):
            for j in range(len(selected_assets)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        st.pyplot(fig)
        footnote("Lower correlations = more diversification benefit available. "
                 "Optimizers exploit exactly this structure.")

# ============================================================
# TAB 2 — EFFICIENT FRONTIER + CML
# ============================================================
with tabs[1]:
    st.subheader("The Markowitz efficient frontier and the Capital Market Line")
    er = pe.annualize_rets(data, 12)
    cov = pe.COV_METHODS[cov_method](data)

    n_points = 40
    target_rs = np.linspace(er.min(), er.max(), n_points)
    frontier_vols = [pe.portfolio_vol(pe._minimize_vol(tr, er.values, cov.values), cov.values)
                     for tr in target_rs]

    markers = {
        "EW": (pe.weights_ew(data), "gold", "o"),
        "GMV": (pe.weights_gmv(data, cov_method=cov_method), "navy", "s"),
        "MSR": (pe.weights_msr(data, riskfree_rate=riskfree_rate, cov_method=cov_method), "green", "^"),
        "MDP": (pe.weights_max_diversification(data, cov_method=cov_method), "teal", "P"),
        "ERC": (pe.weights_risk_parity(data, cov_method=cov_method), "purple", "D"),
        "HRP": (pe.weights_hrp(data, cov_method=cov_method), "crimson", "*"),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frontier_vols, target_rs, "b.-", label="Efficient Frontier", alpha=0.7)

    # Capital Market Line: from (0, rf) through the tangency (MSR) portfolio
    w_msr = markers["MSR"][0]
    r_msr = pe.portfolio_return(w_msr, er.values)
    v_msr = pe.portfolio_vol(w_msr, cov.values)
    cml_x = np.linspace(0, max(frontier_vols) * 1.05, 50)
    cml_slope = (r_msr - riskfree_rate) / v_msr if v_msr > 0 else 0
    ax.plot(cml_x, riskfree_rate + cml_slope * cml_x, "g--", alpha=0.7,
            label=f"Capital Market Line (Sharpe={cml_slope:.2f})")
    ax.scatter([0], [riskfree_rate], color="black", marker="o", s=40, zorder=5)
    ax.annotate("Rf", (0, riskfree_rate), textcoords="offset points", xytext=(8, -4), fontsize=9)

    for label, (w, color, marker) in markers.items():
        r = pe.portfolio_return(w, er.values)
        v = pe.portfolio_vol(w, cov.values)
        ax.scatter([v], [r], color=color, marker=marker, s=130, label=label, zorder=6, edgecolor="white")

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_xlim(left=0)
    ax.set_title(f"Efficient Frontier + CML — {cov_method} covariance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    footnote(
        "The frontier is the set of minimum-variance portfolios for each return level. "
        "The CML runs from the risk-free rate through the tangency (Max-Sharpe) portfolio; its slope IS the best achievable Sharpe ratio. "
        "Every strategy marker is plotted at its in-sample risk/return — but note these are IN-SAMPLE; the Backtest tab shows the honest out-of-sample story."
    )

    st.markdown("**Weights implied by each strategy (in-sample)**")
    weight_table = pd.DataFrame({label: w for label, (w, _, _) in markers.items()}, index=selected_assets)
    st.dataframe((weight_table * 100).round(1), use_container_width=True)

# ============================================================
# TAB 3 — STRATEGY LAB
# ============================================================
with tabs[2]:
    st.subheader("Strategy Lab — pick one, see its weights and read why it behaves that way")
    strat = st.selectbox("Strategy", list(pe.STRATEGIES.keys()), index=5)
    w = pe.STRATEGIES[strat](data, cov_method=cov_method, riskfree_rate=riskfree_rate)
    w = np.clip(w, 0, None); w = w / w.sum()
    wser = pd.Series(w, index=selected_assets).sort_values(ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(wser.index[::-1], wser.values[::-1] * 100, color="steelblue")
        ax.set_xlabel("Weight (%)")
        ax.set_title(f"{strat} weights")
        for i, v in enumerate(wser.values[::-1]):
            ax.text(v * 100 + 0.3, i, f"{v*100:.1f}%", va="center", fontsize=8)
        st.pyplot(fig)
        # in-sample risk/return
        r = pe.portfolio_return(w, pe.annualize_rets(data, 12).values)
        v = pe.portfolio_vol(w, pe.COV_METHODS[cov_method](data).values)
        st.metric("In-sample annualized return", f"{r*100:.1f}%")
        st.metric("In-sample annualized vol", f"{v*100:.1f}%")
    with c2:
        strategy_card(strat)

    st.markdown("---")
    st.markdown("**How concentrated is each strategy?** (effective number of holdings = 1 / Σwᵢ²)")
    conc_rows = []
    for name, fn in pe.STRATEGIES.items():
        ww = fn(data, cov_method=cov_method, riskfree_rate=riskfree_rate)
        ww = np.clip(ww, 0, None); ww = ww / ww.sum()
        enb = 1 / np.sum(ww ** 2)
        conc_rows.append({"Strategy": name, "Effective # holdings": round(enb, 1),
                          "Max weight": f"{ww.max()*100:.0f}%"})
    st.dataframe(pd.DataFrame(conc_rows).set_index("Strategy"), use_container_width=True)
    footnote(
        "Effective number of holdings near N (here up to "
        f"{len(selected_assets)}) means well spread; near 1 means concentrated. "
        "EW is always maximally spread; MSR is usually the most concentrated because it chases the single best in-sample bet."
    )

# ============================================================
# TAB 4 — BACKTEST
# ============================================================
with tabs[3]:
    st.subheader("Walk-forward, out-of-sample backtest (net of costs)")
    st.caption(
        f"Weights re-estimated every {rebalance_every} months on a {window_type} "
        f"{'window of ' + str(window) + ' months' if window_type=='rolling' else 'window (min ' + str(window) + ' months)'}, "
        f"then applied to the FOLLOWING month's realised returns. No look-ahead. "
        f"Transaction cost: {tc_bps} bps one-way on turnover."
    )

    results = {}
    for name, fn in pe.STRATEGIES.items():
        try:
            res = pe.backtest_strategy(data, fn, window=window, rebalance_every=rebalance_every,
                                       cov_method=cov_method, riskfree_rate=riskfree_rate,
                                       transaction_cost_bps=tc_bps, window_type=window_type)
            results[name] = res
        except ValueError as e:
            st.error(f"{name}: {e}")

    if results:
        # Comparison table
        rows = []
        for name, res in results.items():
            r = pe.wealth_to_returns(res["wealth"])
            s = pe.summary_stats(r, riskfree_rate=riskfree_rate)
            rows.append({
                "Strategy": name,
                "Ann Return": f"{s['Annualized Return']*100:.1f}%",
                "Ann Vol": f"{s['Annualized Vol']*100:.1f}%",
                "Sharpe": round(s["Sharpe Ratio"], 2),
                "Sortino": round(s["Sortino Ratio"], 2),
                "Calmar": round(s["Calmar Ratio"], 2),
                "Max DD": f"{s['Max Drawdown']*100:.0f}%",
                "Avg Turnover": round(res["turnover"][res["turnover"] > 0].mean(), 2),
                "Cost Drag": f"{res['total_cost_drag']*100:.2f}%",
                "Final ₹": f"{res['wealth'].iloc[-1]:,.0f}",
            })
        comp = pd.DataFrame(rows).set_index("Strategy")
        st.dataframe(comp, use_container_width=True)

        # Wealth curves
        fig, ax = plt.subplots(figsize=(11, 5))
        for name, res in results.items():
            ax.plot(res["wealth"].index.to_timestamp(), res["wealth"].values, label=name, linewidth=1.6)
        ax.set_ylabel("Wealth (₹, start = 1000)")
        ax.set_title("Cumulative wealth, net of costs — all strategies")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # Auto-generated commentary
        best_sharpe = max(results, key=lambda n: pe.summary_stats(pe.wealth_to_returns(results[n]["wealth"]), riskfree_rate=riskfree_rate)["Sharpe Ratio"])
        shallowest_dd = min(results, key=lambda n: abs(pe.summary_stats(pe.wealth_to_returns(results[n]["wealth"]), riskfree_rate=riskfree_rate)["Max Drawdown"]))
        st.markdown(
            f"**Reading the results.** On this run, **{best_sharpe}** has the best risk-adjusted "
            f"return (Sharpe) and **{shallowest_dd}** has the shallowest drawdown. "
            "A recurring pattern in Indian sector data: the strategies that DON'T rely on forecasting "
            "expected returns (HRP, Risk Parity, GMV, Equal Weight) tend to beat Max Sharpe out-of-sample, "
            "because historical mean returns are too noisy to optimize on directly — the textbook case for robust methods."
        )
        footnote(
            "Sharpe is sensitive to the risk-free slider; Sortino and Calmar less so. "
            "Judge strategies on the RELATIVE ranking, which is stable, more than the absolute numbers. "
            "Higher turnover strategies lose more to the cost drag — visible in the two right-hand columns."
        )

# ============================================================
# TAB 5 — REGIME ANALYSIS
# ============================================================
with tabs[4]:
    st.subheader("Market regimes (unsupervised ML: Gaussian mixture)")
    st.caption(
        "A 2-state Gaussian Mixture Model is fit to a benchmark return series, "
        "splitting history into a calm/bull regime and a turbulent/bear regime by "
        "their return-and-volatility signature. This is the Course 3 regime-analysis idea in miniature."
    )
    bench_options = benchmarks.columns.tolist()
    bench_pick = st.selectbox("Benchmark to detect regimes on", bench_options,
                              index=bench_options.index("Nifty50") if "Nifty50" in bench_options else 0)
    n_reg = st.radio("Number of regimes", [2, 3], index=0, horizontal=True)

    regime_df, gmm = pe.detect_regimes(benchmarks[bench_pick], n_regimes=n_reg)
    stats_tbl = pe.regime_stats(regime_df)
    st.dataframe(stats_tbl, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_reg))
    name_order = regime_df.groupby("Regime Name")["Return"].mean().sort_values().index.tolist()
    color_map = {nm: colors[i] for i, nm in enumerate(name_order)}
    ts = regime_df.index.to_timestamp()
    wealth_bench = 1000 * (1 + regime_df["Return"]).cumprod()
    for nm in name_order:
        mask = regime_df["Regime Name"] == nm
        ax.scatter(ts[mask], wealth_bench[mask], s=14, color=color_map[nm], label=nm)
    ax.plot(ts, wealth_bench, color="gray", linewidth=0.6, alpha=0.5)
    ax.set_ylabel(f"{bench_pick} wealth (₹, start=1000)")
    ax.set_title(f"{bench_pick} coloured by detected regime")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    footnote(
        "The turbulent regime clusters the crash months (e.g. March 2020) and carries far higher "
        "volatility and deeper worst-months. In practice you'd de-risk (shift toward GMV / CPPI / cash) "
        "when the model flags a turbulent state. Caveat: regimes are labelled in-sample here; a live system "
        "would classify the current month from the fitted model without peeking at the future."
    )

# ============================================================
# TAB 6 — CPPI
# ============================================================
with tabs[5]:
    st.subheader("CPPI — dynamic downside insurance")
    st.caption(
        "Constant Proportion Portfolio Insurance keeps wealth above a floor by "
        "allocating m × (wealth − floor) to the risky strategy and the rest to a "
        "safe asset. Optionally the floor ratchets up with the peak (drawdown-constrained / TIPP)."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cppi_strat = st.selectbox("Risky strategy", list(pe.STRATEGIES.keys()), index=5, key="cppi_strat")
    with c2:
        cppi_m = st.slider("Multiplier m", 1.0, 6.0, 3.0, 0.5)
    with c3:
        cppi_floor = st.slider("Floor (% of start)", 0.50, 0.95, 0.80, 0.05)
    with c4:
        use_dd = st.checkbox("Ratcheting floor (TIPP)", value=False)
    dd_limit = (1 - cppi_floor) if use_dd else None

    res = pe.backtest_strategy(data, pe.STRATEGIES[cppi_strat], window=window,
                               rebalance_every=rebalance_every, cov_method=cov_method,
                               riskfree_rate=riskfree_rate, transaction_cost_bps=tc_bps,
                               window_type=window_type)
    risky_r = pe.wealth_to_returns(res["wealth"]).to_frame(cppi_strat)
    cppi = pe.run_cppi(risky_r, m=cppi_m, start=1000, floor=cppi_floor,
                       riskfree_rate=riskfree_rate, drawdown_limit=dd_limit)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(cppi["Wealth"].index.to_timestamp(), cppi["Wealth"][cppi_strat], label="CPPI wealth", linewidth=2)
    ax.plot(cppi["Risky Wealth"].index.to_timestamp(), cppi["Risky Wealth"][cppi_strat],
            label=f"{cppi_strat} (uninsured)", linewidth=1.5, linestyle="--")
    ax.plot(cppi["Floor Value"].index.to_timestamp(), cppi["Floor Value"][cppi_strat],
            label="Floor", linewidth=1.2, linestyle=":", color="red")
    ax.set_ylabel("Wealth (₹, start=1000)")
    ax.set_title(f"CPPI (m={cppi_m}, floor={cppi_floor*100:.0f}%{', ratcheting' if use_dd else ''})")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(11, 2.8))
    ax2.plot(cppi["Risky Allocation"].index.to_timestamp(), cppi["Risky Allocation"][cppi_strat] * 100,
             color="green", linewidth=1.8)
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Risky alloc (%)"); ax2.grid(alpha=0.3)
    ax2.set_title("CPPI dynamically de-risks as the cushion shrinks")
    st.pyplot(fig2)

    c1, c2 = st.columns(2)
    c1.metric("CPPI final wealth", f"₹{cppi['Wealth'][cppi_strat].iloc[-1]:,.0f}")
    c2.metric("Uninsured final wealth", f"₹{cppi['Risky Wealth'][cppi_strat].iloc[-1]:,.0f}")
    footnote(
        "Pros: hard floor protection, participates in upside via the multiplier. "
        "Cons: in a sharp V-shaped crash (like March 2020) CPPI can 'lock in' losses by de-risking at "
        "the bottom and miss the rebound — the classic cash-lock / gap-risk criticism. Higher m = more "
        "upside but more gap risk. A higher floor is safer but caps growth."
    )

# ============================================================
# TAB 7 — ALM / PENSION
# ============================================================
with tabs[6]:
    st.subheader("Asset-Liability Management — the pension / LDI lens")
    st.caption(
        "Course 1 Week 4. A pension fund cares about its liabilities (future payouts), not just asset "
        "returns. Two ideas: the funding ratio (are we solvent at today's rates?) and duration matching "
        "(immunize the liability against rate moves)."
    )

    st.markdown("#### 1. Funding ratio vs. interest rates")
    c1, c2, c3 = st.columns(3)
    with c1:
        asset_value = st.number_input("Current asset value (₹)", 100.0, 1e9, 500.0, step=50.0)
    with c2:
        liab_amount = st.number_input("Annual liability payout (₹)", 1.0, 1e9, 100.0, step=10.0)
    with c3:
        liab_years = st.slider("Liability paid in years", 1, 30, [5, 15],
                               help="Range of future years the fund owes payouts")
    years = list(range(liab_years[0], liab_years[1] + 1))
    liabilities = pd.Series(liab_amount, index=years)
    rates = np.linspace(0.01, 0.10, 30)
    fr = pd.Series({r: pe.funding_ratio(pd.Series([asset_value], index=[0]), liabilities, r) for r in rates})

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rates * 100, fr.values, linewidth=2, color="darkblue")
    ax.axhline(1.0, color="red", linestyle="--", label="Fully funded (ratio = 1)")
    ax.axvline(riskfree_rate * 100, color="green", linestyle=":", label=f"Current rate ({riskfree_rate*100:.1f}%)")
    ax.set_xlabel("Discount rate (%)"); ax.set_ylabel("Funding ratio")
    ax.set_title("Funding ratio rises with rates (liabilities discounted more)")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)
    fr_now = pe.funding_ratio(pd.Series([asset_value], index=[0]), liabilities, riskfree_rate)
    st.metric(f"Funding ratio at {riskfree_rate*100:.1f}%", f"{fr_now:.2f}",
              delta="surplus" if fr_now >= 1 else "shortfall")
    footnote(
        "This is why falling rates are a pension fund's nightmare: liabilities balloon (discounted less) "
        "faster than a typical asset mix, so the funding ratio drops. LDI hedges exactly this."
    )

    st.markdown("---")
    st.markdown("#### 2. Duration matching (immunization)")
    c1, c2, c3 = st.columns(3)
    with c1:
        short_mat = st.slider("Short bond maturity (yrs)", 1, 10, 3)
    with c2:
        long_mat = st.slider("Long bond maturity (yrs)", 10, 30, 20)
    with c3:
        disc = st.slider("Flat discount rate", 0.02, 0.10, riskfree_rate, 0.005, key="dur_disc")
    liab_cf = pe.bond_cash_flows(sum(years) / len(years), coupon_rate=0.0, coupons_per_year=1)
    # target liability = the pension payout stream itself
    liab_cf = liabilities.copy()
    short_cf = pe.bond_cash_flows(short_mat, coupon_rate=0.05, coupons_per_year=1)
    long_cf = pe.bond_cash_flows(long_mat, coupon_rate=0.05, coupons_per_year=1)
    d_liab = pe.macaulay_duration(liab_cf, disc)
    d_short = pe.macaulay_duration(short_cf, disc)
    d_long = pe.macaulay_duration(long_cf, disc)
    w_short = pe.match_durations(liab_cf, short_cf, long_cf, disc)
    w_short = float(np.clip(w_short, 0, 1))

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Liability duration", f"{d_liab:.1f} yrs")
    cc2.metric("Weight in short bond", f"{w_short*100:.0f}%")
    cc3.metric("Weight in long bond", f"{(1-w_short)*100:.0f}%")
    st.markdown(
        f"To immunize a liability with **{d_liab:.1f}-year** duration using a "
        f"**{short_mat}yr** bond (duration {d_short:.1f}) and a **{long_mat}yr** bond "
        f"(duration {d_long:.1f}), hold **{w_short*100:.0f}%** short / **{(1-w_short)*100:.0f}%** long — "
        "so the blended duration matches the liability and small rate moves hit assets and liabilities equally."
    )
    footnote(
        "This is the mechanical core of a liability-hedging portfolio. A full pension solution splits capital "
        "between this hedging portfolio and a return-seeking portfolio (the equity strategies in the other tabs) — "
        "that split is the 'risk budget'. Building that combined LDI + growth engine is the natural next module."
    )

# ============================================================
# TAB 8 — NOTES
# ============================================================
with tabs[7]:
    st.subheader("Methodology, data, and honest caveats")
    st.markdown(
        """
**What this tool implements (mapped to the EDHEC specialization):**

- **Course 1** — annualized return/vol, Sharpe/Sortino/Calmar, VaR/CVaR, drawdown; the mean-variance
  efficient frontier and Capital Market Line; CPPI/TIPP insurance; ALM (funding ratio, duration matching).
- **Course 2** — robust covariance (Ledoit-Wolf shrinkage, constant-correlation, EWMA forecast);
  Black-Litterman equilibrium (implied returns instead of noisy historical means).
- **Course 3** — Hierarchical Risk Parity (clustering-based ML allocation); regime analysis via a
  Gaussian mixture model.

**Backtesting discipline:** every backtest is walk-forward and strictly out-of-sample — weights at month *t*
use only data before *t*, then get applied to *t*'s realised return. Transaction costs are charged on turnover.
Rolling vs. expanding windows are both available.
        """
    )
    st.markdown("**Covariance estimator notes:**")
    for k, v in pe.COV_INFO.items():
        st.markdown(f"- **{k}** — {v}")
    st.markdown(
        """
**What still needs verification (be honest with your instructor):**

- **Data provenance.** The NSE sector/benchmark returns come from Yahoo Finance index tickers, not an
  official NSE feed. Spot-check a few known months (e.g. March 2020) against NSE factsheets before citing exact numbers.
- **Sharpe vs. the risk-free rate.** At a 6.5% Indian G-Sec Rf, Sharpe ratios look low versus US figures
  quoted at 2%. That is an apples-to-oranges artifact — report the relative ranking and also show Sortino/Calmar.
- **Costs are a flat bps assumption**, not a real cost model (impact, bid-ask, STT/GST detail).
- **Regimes and in-sample markers** on the frontier are illustrative; the honest performance numbers are on the Backtest tab.
        """
    )
    st.caption("Built on the EDHEC 'Investment Management with Python and Machine Learning' methodology, "
               "applied to Indian NSE indices.")
