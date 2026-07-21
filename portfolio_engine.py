"""
portfolio_engine.py
--------------------
Production-grade extension of edhec_risk_kit.py for the Indian Multi-Strategy
Portfolio & Risk Engine (EDHEC "Investment Management with Python and ML" project,
Paterson Securities internship).

Design notes:
- No notebook-only imports (ipywidgets / IPython.display) so this is safe to
  deploy headless on Streamlit Community Cloud.
- All return series are expected in DECIMAL form (0.05 = 5%), monthly frequency,
  pandas PeriodIndex or DatetimeIndex — matches edhec_risk_kit.py conventions.
- Strategies are exposed as functions with signature (returns_window: DataFrame) -> np.ndarray
  of weights aligned to returns_window.columns, so they plug directly into backtest_strategy().

Strategies implemented:
    1. Equal Weight (EW)
    2. Global Minimum Variance (GMV)
    3. Max Sharpe / Tangency (MSR)
    4. Equal Risk Contribution / Risk Parity (ERC)
    5. Hierarchical Risk Parity (HRP) -- Lopez de Prado (2016)
    6. Any of the above with Ledoit-Wolf shrinkage covariance
    7. Any of the above with EWMA (RiskMetrics) forecast covariance
    8. CPPI insurance overlay (path-dependent, run separately via run_cppi)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_returns(path, pct_format=True, date_col="Date"):
    """
    Load a returns CSV in the ind_in_m_*.csv format (Date column, one column
    per asset, values in PERCENT e.g. 11.11 = 11.11%).
    Returns a DataFrame of DECIMAL returns indexed by monthly PeriodIndex.
    """
    df = pd.read_csv(path, index_col=date_col, parse_dates=False)
    df.index = pd.to_datetime(df.index, format="%Y-%m").to_period("M")
    if pct_format:
        df = df / 100
    return df


# ============================================================
# 2. CORE STATS (self-contained, mirrors edhec_risk_kit.py)
# ============================================================

def annualize_rets(r, periods_per_year=12):
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r, periods_per_year=12):
    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year=12):
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def sortino_ratio(r, riskfree_rate, periods_per_year=12):
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    downside = excess_ret[excess_ret < 0]
    downside_dev = downside.std(ddof=0) * (periods_per_year ** 0.5)
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    if downside_dev == 0 or pd.isna(downside_dev):
        return np.nan
    return ann_ex_ret / downside_dev


def drawdown(return_series: pd.Series):
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns})


def max_drawdown(return_series: pd.Series):
    return drawdown(return_series).Drawdown.min()


def calmar_ratio(r, periods_per_year=12):
    ann_r = annualize_rets(r, periods_per_year)
    mdd = abs(max_drawdown(r))
    if mdd == 0:
        return np.nan
    return ann_r / mdd


def skewness(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    return (demeaned_r ** 3).mean() / sigma_r ** 3


def kurtosis(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    return (demeaned_r ** 4).mean() / sigma_r ** 4


def var_historic(r, level=5):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    raise TypeError("Expected r to be Series or DataFrame")


def cvar_historic(r, level=5):
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    raise TypeError("Expected r to be a Series or DataFrame")


def summary_stats(r, riskfree_rate=0.065, periods_per_year=12):
    """riskfree_rate default = India 10Y G-Sec proxy, override as needed."""
    ann_r = r.aggregate(annualize_rets, periods_per_year=periods_per_year) if isinstance(r, pd.DataFrame) else annualize_rets(r, periods_per_year)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=periods_per_year) if isinstance(r, pd.DataFrame) else annualize_vol(r, periods_per_year)
    if isinstance(r, pd.DataFrame):
        ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
        sortino = r.aggregate(sortino_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
        calmar = r.aggregate(calmar_ratio, periods_per_year=periods_per_year)
        dd = r.aggregate(lambda x: drawdown(x).Drawdown.min())
        skew = r.aggregate(skewness)
        kurt = r.aggregate(kurtosis)
        var5 = r.aggregate(var_historic)
        cvar5 = r.aggregate(cvar_historic)
    else:
        ann_sr = sharpe_ratio(r, riskfree_rate, periods_per_year)
        sortino = sortino_ratio(r, riskfree_rate, periods_per_year)
        calmar = calmar_ratio(r, periods_per_year)
        dd = drawdown(r).Drawdown.min()
        skew = skewness(r)
        kurt = kurtosis(r)
        var5 = var_historic(r)
        cvar5 = cvar_historic(r)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": ann_sr,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": dd,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Historic VaR (5%)": var5,
        "Historic CVaR (5%)": cvar5,
    }) if isinstance(r, pd.DataFrame) else pd.Series({
        "Annualized Return": ann_r, "Annualized Vol": ann_vol, "Sharpe Ratio": ann_sr,
        "Sortino Ratio": sortino, "Calmar Ratio": calmar, "Max Drawdown": dd,
        "Skewness": skew, "Kurtosis": kurt, "Historic VaR (5%)": var5, "Historic CVaR (5%)": cvar5,
    })


# ============================================================
# 3. COVARIANCE ESTIMATORS
# ============================================================

def sample_cov(r, periods_per_year=12):
    return r.cov() * periods_per_year


def shrinkage_cov(r, periods_per_year=12):
    """
    Ledoit-Wolf shrinkage covariance. Shrinks the noisy sample covariance
    toward a structured (scaled identity) target -- reduces estimation error,
    especially important with short Indian sector histories (~175 months).
    Falls back to sample covariance if scikit-learn isn't available.
    """
    if not _HAS_SKLEARN:
        return sample_cov(r, periods_per_year)
    lw = LedoitWolf().fit(r.values)
    cov = pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)
    return cov * periods_per_year


def ewma_cov(r, lam=0.94, periods_per_year=12):
    """
    RiskMetrics-style EWMA covariance forecast. This is the "predictive"
    layer: instead of an equal-weighted historical covariance, recent months
    get exponentially more weight, so the forecast reacts to changing
    volatility regimes (e.g. COVID crash, rate-hike periods).
    sigma_t^2 = lam * sigma_(t-1)^2 + (1-lam) * r_(t-1)^2
    """
    assets = r.columns
    T = len(r)
    demeaned = (r - r.mean()).values
    cov = np.cov(demeaned, rowvar=False)  # seed with sample cov
    for t in range(T):
        x = demeaned[t].reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (x @ x.T)
    cov_df = pd.DataFrame(cov, index=assets, columns=assets)
    return cov_df * periods_per_year


def ewma_vol_forecast(r, lam=0.94, periods_per_year=12):
    """Per-asset EWMA annualized volatility forecast (diagonal of ewma_cov)."""
    cov = ewma_cov(r, lam=lam, periods_per_year=periods_per_year)
    return np.sqrt(np.diag(cov))


def constant_correlation_cov(r, periods_per_year=12):
    """
    Elton-Gruber constant-correlation covariance estimate. Replaces the full
    correlation matrix with a single average pairwise correlation, keeping each
    asset's own volatility. A structured, low-noise estimator: it throws away
    unreliable individual pairwise correlations (which are hard to estimate from
    short samples) in favour of one robust average, then rebuilds the covariance.
    """
    corr = r.corr().values
    n = corr.shape[0]
    off_diag = (corr.sum() - np.trace(corr)) / (n * (n - 1))
    ccor = np.full_like(corr, off_diag)
    np.fill_diagonal(ccor, 1.0)
    sd = r.std().values
    cov = ccor * np.outer(sd, sd)
    return pd.DataFrame(cov, index=r.columns, columns=r.columns) * periods_per_year


COV_METHODS = {
    "sample": sample_cov,
    "shrinkage": shrinkage_cov,
    "constant_corr": constant_correlation_cov,
    "ewma": ewma_cov,
}


# ============================================================
# 4. PORTFOLIO MATH
# ============================================================

def portfolio_return(weights, returns):
    return np.dot(np.asarray(weights).T, np.asarray(returns))


def portfolio_vol(weights, covmat):
    w = np.asarray(weights)
    return (w.T @ np.asarray(covmat) @ w) ** 0.5


# ============================================================
# 5. WEIGHT / STRATEGY FUNCTIONS
#    All take a returns_window DataFrame -> return np.ndarray of weights
# ============================================================

def weights_ew(returns_window, **kwargs):
    n = returns_window.shape[1]
    return np.repeat(1 / n, n)


def _minimize_vol(target_return, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {"type": "eq", "args": (er,),
                         "fun": lambda w, er: target_return - portfolio_return(w, er)}
    weights_sum_to_1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    results = minimize(portfolio_vol, init_guess, args=(cov,), method="SLSQP",
                        options={"disp": False}, constraints=(return_is_target, weights_sum_to_1),
                        bounds=bounds)
    return results.x


def weights_gmv(returns_window, cov_method="sample", **kwargs):
    cov = COV_METHODS[cov_method](returns_window)
    n = cov.shape[0]
    er = np.repeat(1, n)  # direction irrelevant for GMV
    return weights_msr(returns_window, riskfree_rate=0, er_override=er, cov_method=cov_method)


def weights_msr(returns_window, riskfree_rate=0.065, cov_method="sample", er_override=None, **kwargs):
    cov = COV_METHODS[cov_method](returns_window)
    er = er_override if er_override is not None else annualize_rets(returns_window, 12).values
    n = len(er)
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def neg_sharpe(w, rf, er, cov):
        r = portfolio_return(w, er) - rf
        vol = portfolio_vol(w, cov)
        return -r / vol if vol > 0 else 0.0

    results = minimize(neg_sharpe, init_guess, args=(riskfree_rate, er, cov.values), method="SLSQP",
                        options={"disp": False}, constraints=(weights_sum_to_1,), bounds=bounds)
    return results.x


def weights_risk_parity(returns_window, cov_method="sample", **kwargs):
    """
    Equal Risk Contribution (ERC) / Risk Parity: find weights such that every
    asset contributes the same share of total portfolio risk. Long-only.
    """
    cov = COV_METHODS[cov_method](returns_window).values
    n = cov.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((1e-4, 1.0),) * n
    weights_sum_to_1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def risk_contributions(w, cov):
        total_vol = portfolio_vol(w, cov)
        marginal_contrib = cov @ w
        return w * marginal_contrib / total_vol

    def objective(w, cov):
        rc = risk_contributions(w, cov)
        target = rc.mean()
        return np.sum((rc - target) ** 2)

    results = minimize(objective, init_guess, args=(cov,), method="SLSQP",
                        options={"disp": False, "maxiter": 1000},
                        constraints=(weights_sum_to_1,), bounds=bounds)
    w = results.x
    return w / w.sum()


def weights_max_diversification(returns_window, cov_method="sample", **kwargs):
    """
    Most Diversified Portfolio (Choueifaty-Coignard). Maximises the
    "diversification ratio" = (weighted average of asset vols) / (portfolio vol).
    A ratio of 1 means no diversification benefit; higher means the portfolio
    volatility is much lower than the average of its parts. Long-only.
    """
    cov = COV_METHODS[cov_method](returns_window).values
    vols = np.sqrt(np.diag(cov))
    n = cov.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def neg_div_ratio(w, vols, cov):
        port_vol = portfolio_vol(w, cov)
        weighted_avg_vol = np.dot(w, vols)
        return -weighted_avg_vol / port_vol if port_vol > 0 else 0.0

    results = minimize(neg_div_ratio, init_guess, args=(vols, cov), method="SLSQP",
                        options={"disp": False, "maxiter": 1000},
                        constraints=(weights_sum_to_1,), bounds=bounds)
    w = np.clip(results.x, 0, None)
    return w / w.sum()


def implied_returns(cov, w_prior, delta=2.5):
    """
    Reverse-optimization (Black-Litterman step 1): back out the expected returns
    that would make the prior weights optimal. pi = delta * cov @ w_prior.
    delta is the market risk-aversion coefficient.
    """
    return delta * cov.dot(w_prior)


def weights_black_litterman(returns_window, cov_method="shrinkage", delta=2.5,
                             riskfree_rate=0.065, w_prior=None, **kwargs):
    """
    Black-Litterman with NO active views -> returns the reverse-optimized
    equilibrium portfolio, i.e. the prior. The value here is pedagogical and
    practical: instead of feeding NOISY historical mean returns into the
    optimizer (which is what makes plain Max Sharpe unstable), BL starts from
    the returns *implied* by a sensible prior allocation. With no views, the
    equilibrium simply reproduces the prior weights -- a deliberately robust,
    stable allocation. Views can be layered on later.
    """
    cov = COV_METHODS[cov_method](returns_window)
    n = cov.shape[0]
    if w_prior is None:
        w_prior = pd.Series(np.repeat(1 / n, n), index=returns_window.columns)
    else:
        w_prior = pd.Series(w_prior, index=returns_window.columns)
    pi = implied_returns(cov, w_prior, delta=delta)
    # With no views, posterior expected returns == equilibrium implied returns.
    # Optimal weights under mean-variance with these returns recover the prior,
    # but we run the MSR optimizer on pi for a clean, constrained long-only result.
    return weights_msr(returns_window, riskfree_rate=riskfree_rate,
                        cov_method=cov_method, er_override=pi.values)


def _get_ivp(cov):
    """Inverse-variance portfolio (used inside HRP recursive bisection)."""
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _get_cluster_var(cov, items):
    cov_slice = cov.loc[items, items]
    w = _get_ivp(cov_slice.values).reshape(-1, 1)
    return (w.T @ cov_slice.values @ w)[0, 0]


def _get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _get_rec_bisection(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in
                   ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            var0 = _get_cluster_var(cov, c_items0)
            var1 = _get_cluster_var(cov, c_items1)
            alpha = 1 - var0 / (var0 + var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w


def weights_hrp(returns_window, cov_method="sample", linkage_method="single", **kwargs):
    """
    Hierarchical Risk Parity (Lopez de Prado, 2016). Uses hierarchical
    clustering on the correlation-distance matrix to group similar assets,
    then allocates risk via recursive bisection down the tree -- no
    matrix inversion required, so it's numerically stable even with a
    short, noisy Indian-sector covariance matrix.
    """
    cov = COV_METHODS[cov_method](returns_window)
    corr = returns_window.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    link = linkage(squareform(dist.values, checks=False), method=linkage_method)
    sort_ix = _get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()
    hrp_w = _get_rec_bisection(cov, sort_ix)
    # reorder back to original column order
    return hrp_w.reindex(returns_window.columns).values


STRATEGIES = {
    "Equal Weight": weights_ew,
    "Global Minimum Variance (GMV)": weights_gmv,
    "Max Sharpe (MSR)": weights_msr,
    "Max Diversification (MDP)": weights_max_diversification,
    "Risk Parity (ERC)": weights_risk_parity,
    "Hierarchical Risk Parity (HRP)": weights_hrp,
    "Black-Litterman (equilibrium)": weights_black_litterman,
}


# ============================================================
# STRATEGY & COVARIANCE METADATA (footnotes for the dashboard)
# ============================================================

STRATEGY_INFO = {
    "Equal Weight": {
        "idea": "Put 1/N in every asset. No estimation of returns or risk at all.",
        "pros": "Impossible to overfit; a famously hard benchmark to beat (DeMiguel, Garlappi & Uppal 2009). Zero estimation error.",
        "cons": "Ignores risk entirely — loads just as much into a 36%-vol PSU Bank basket as a 15%-vol FMCG basket. Higher turnover than cap-weight.",
        "real_world": "The honest baseline every institutional allocator measures against. If a 'smart' strategy can't beat 1/N out-of-sample, it isn't smart.",
    },
    "Global Minimum Variance (GMV)": {
        "idea": "Choose weights that minimise portfolio volatility, using only the covariance matrix (no return forecasts).",
        "pros": "Avoids the noisiest input (expected returns). Historically delivers strong risk-adjusted returns and the shallowest drawdowns of the mean-variance family.",
        "cons": "Concentrates in low-vol assets; sensitive to covariance estimation error (helped a lot by shrinkage). Can ignore attractive but volatile sectors.",
        "real_world": "The workhorse of 'minimum volatility' / 'low-vol' smart-beta funds. Widely used precisely because it sidesteps return forecasting.",
    },
    "Max Sharpe (MSR)": {
        "idea": "The tangency portfolio — maximise (return − rf) / volatility. Needs BOTH expected returns and covariance.",
        "pros": "Theoretically optimal on the efficient frontier; the point the Capital Market Line touches.",
        "cons": "Depends on historical mean returns, which are extremely noisy. Tends to make big, unstable bets and often underperforms out-of-sample — 'error maximization' (Michaud).",
        "real_world": "Textbook-optimal, practically fragile. Its instability is the reason robust methods (shrinkage, Black-Litterman, HRP) exist.",
    },
    "Max Diversification (MDP)": {
        "idea": "Maximise the diversification ratio: weighted-average asset vol divided by portfolio vol.",
        "pros": "Explicitly hunts for diversification benefit; tends to spread risk across uncorrelated exposures rather than piling into one low-vol block.",
        "cons": "Still covariance-dependent; can tilt toward high-vol assets that happen to be uncorrelated. No return input.",
        "real_world": "Commercialised by TOBAM as 'Anti-Benchmark' funds. A genuine institutional product, not just an academic idea.",
    },
    "Risk Parity (ERC)": {
        "idea": "Size positions so each asset contributes the SAME amount of risk to the portfolio (Equal Risk Contribution).",
        "pros": "Well-diversified by construction; no return forecasts; more stable than MSR. Naturally down-weights the wild sectors.",
        "cons": "Assumes equal risk = good, which isn't always true; ignores expected returns; in levered versions (Bridgewater-style) it adds leverage risk.",
        "real_world": "The basis of Bridgewater's 'All Weather' and a whole category of risk-parity funds — one of the most successful practical allocation ideas of the last 20 years.",
    },
    "Hierarchical Risk Parity (HRP)": {
        "idea": "Machine-learning method (Lopez de Prado 2016): cluster assets by correlation, then split risk down the tree via recursive bisection. No matrix inversion.",
        "pros": "Numerically stable where MSR/GMV break down (near-singular covariance); respects the correlation structure (e.g. keeps the two bank sectors together). Robust to noisy, short samples like ours.",
        "cons": "Newer, less intuitive to explain; results depend on the clustering/linkage choice; still no return forecast.",
        "real_world": "The flagship 'ML for portfolio construction' technique — exactly the Course 3, Week 3 material, and increasingly used at quant funds.",
    },
    "Black-Litterman (equilibrium)": {
        "idea": "Reverse-engineer the expected returns implied by a sensible prior allocation, instead of trusting raw historical means. Here we run it with no active views, so it returns the robust equilibrium.",
        "pros": "Fixes Max Sharpe's core weakness — replaces noisy historical means with stable, market-implied returns. Views can be blended in with confidence levels.",
        "cons": "With no views it reduces to the prior (here, an equal-risk-ish equilibrium); the choice of prior and risk-aversion delta matters; full power only shows with well-formed views.",
        "real_world": "Developed at Goldman Sachs; the industry-standard way institutional desks combine a market prior with analyst views.",
    },
}

COV_INFO = {
    "sample": "Plain historical covariance. Unbiased but noisy — with 11 sectors and ~175 months, many pairwise correlations are estimated imprecisely.",
    "shrinkage": "Ledoit-Wolf shrinkage pulls the noisy sample matrix toward a structured target. Reduces estimation error; usually the best default for optimization.",
    "constant_corr": "Elton-Gruber: replace all pairwise correlations with one average correlation. Maximally structured, minimally noisy — a robust foil to the sample matrix.",
    "ewma": "Exponentially-weighted (RiskMetrics, λ=0.94). Recent months count more, so it forecasts the CURRENT volatility regime rather than averaging across calm and crisis equally.",
}


# ============================================================
# 6. BACKTEST ENGINE
# ============================================================

def backtest_strategy(returns, strategy_fn, window=36, rebalance_every=1,
                       start=1000, cov_method="sample", riskfree_rate=0.065,
                       transaction_cost_bps=0.0, window_type="rolling"):
    """
    Walk-forward, out-of-sample backtest with transaction costs and turnover.

    - returns: full history of DECIMAL monthly returns (DataFrame)
    - strategy_fn: one of the functions in STRATEGIES
    - window: trailing months used to estimate weights (rolling) or minimum
      months before the first rebalance (expanding)
    - rebalance_every: rebalance frequency in months
    - transaction_cost_bps: one-way cost in basis points applied to turnover at
      each rebalance (e.g. 10 = 0.10% per unit traded). India cash-equity all-in
      costs are roughly 10-30 bps; 10 is a reasonable default for index ETFs.
    - window_type: "rolling" (fixed lookback) or "expanding" (all history to date)

    No look-ahead: weights at month t use only data strictly before t
    (returns.iloc[start:t]); they are then applied to month t's realised return.

    Returns dict with: wealth, gross_wealth (no costs), weights (DataFrame),
    turnover (Series), total_cost_drag (float, wealth fraction lost to costs).
    """
    assets = returns.columns
    dates = returns.index
    if len(dates) <= window:
        raise ValueError(f"Not enough history: need > {window} months, got {len(dates)}")

    tc_rate = transaction_cost_bps / 10000.0
    wealth_val = start
    gross_val = start
    wealth_hist, gross_hist, weight_rows, turnover_rows = [], [], [], []
    prev_w = None

    for t in range(window, len(dates)):
        if prev_w is None or (t - window) % rebalance_every == 0:
            train_start = 0 if window_type == "expanding" else t - window
            train = returns.iloc[train_start:t]
            new_w = strategy_fn(train, cov_method=cov_method, riskfree_rate=riskfree_rate)
            new_w = np.clip(new_w, 0, None)
            s = new_w.sum()
            new_w = new_w / s if s > 0 else np.repeat(1 / len(assets), len(assets))
            prior = prev_w if prev_w is not None else np.zeros_like(new_w)
            turnover = np.abs(new_w - prior).sum()  # sum of |trades|, in [0, 2]
            cost = tc_rate * turnover
            wealth_val *= (1 - cost)
            prev_w = new_w
        else:
            turnover = 0.0
        turnover_rows.append((dates[t], turnover))
        r_t = returns.iloc[t].values
        port_ret = float(np.dot(prev_w, r_t))
        wealth_val *= (1 + port_ret)
        gross_val *= (1 + port_ret)
        wealth_hist.append((dates[t], wealth_val))
        gross_hist.append((dates[t], gross_val))
        weight_rows.append((dates[t], prev_w.copy()))

    wealth = pd.Series(dict(wealth_hist))
    gross_wealth = pd.Series(dict(gross_hist))
    weight_history = pd.DataFrame({d: w for d, w in weight_rows}, index=assets).T
    turnover = pd.Series(dict(turnover_rows))
    total_cost_drag = 1 - (wealth.iloc[-1] / gross_wealth.iloc[-1]) if gross_wealth.iloc[-1] > 0 else 0.0
    return {
        "wealth": wealth,
        "gross_wealth": gross_wealth,
        "weights": weight_history,
        "turnover": turnover,
        "total_cost_drag": total_cost_drag,
    }


def wealth_to_returns(wealth: pd.Series):
    return wealth.pct_change().dropna()


# ============================================================
# 7. CPPI INSURANCE OVERLAY (path-dependent, kept close to edhec_risk_kit.py)
# ============================================================

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.065, drawdown_limit=None):
    """
    CPPI backtest. risky_r must be a DataFrame (single or multi-column).
    drawdown_limit: if set, floor ratchets up with the peak (max-drawdown-constrained CPPI / TIPP-style).
    """
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r)

    if safe_r is None:
        safe_r = pd.DataFrame(riskfree_rate / 12, index=risky_r.index, columns=risky_r.columns)

    account_history = pd.DataFrame(0.0, index=risky_r.index, columns=risky_r.columns)
    risky_w_history = pd.DataFrame(0.0, index=risky_r.index, columns=risky_r.columns)
    cushion_history = pd.DataFrame(0.0, index=risky_r.index, columns=risky_r.columns)
    floorval_history = pd.DataFrame(0.0, index=risky_r.index, columns=risky_r.columns)

    for step in range(n_steps):
        if drawdown_limit is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown_limit)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.clip(risky_w, 0, 1)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value

    risky_wealth = start * (1 + risky_r).cumprod()
    return {
        "Wealth": account_history, "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history, "Risky Allocation": risky_w_history,
        "Floor Value": floorval_history, "m": m, "start": start, "floor": floor,
    }


# ============================================================
# 8. REGIME ANALYSIS (Course 3, Week 4)
# ============================================================

def detect_regimes(benchmark_returns, n_regimes=2, random_state=42):
    """
    Fit a Gaussian Mixture Model on a benchmark return series to classify each
    month into a market regime (e.g. calm/bull vs turbulent/bear). This is a
    lightweight, explainable version of the regime-switching models in Course 3:
    unsupervised ML that separates 'normal' months from 'crisis' months by their
    return/volatility signature.

    Returns a DataFrame with the return, assigned regime label, and a
    human-readable regime name (the lowest-mean regime is tagged 'Turbulent').
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError("scikit-learn is required for regime detection")

    r = benchmark_returns.dropna()
    X = r.values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_regimes, covariance_type="full",
                          random_state=random_state, n_init=5)
    labels = gmm.fit_predict(X)
    means = gmm.means_.flatten()
    # Rank regimes by mean return: lowest = most turbulent/bearish
    order = np.argsort(means)
    if n_regimes == 2:
        names = {order[0]: "Turbulent / Bear", order[1]: "Calm / Bull"}
    else:
        names = {lab: f"Regime {rank+1} (mean {means[lab]*100:.1f}%/m)"
                 for rank, lab in enumerate(order)}
    out = pd.DataFrame({"Return": r.values, "Regime": labels}, index=r.index)
    out["Regime Name"] = out["Regime"].map(names)
    return out, gmm


def regime_stats(regime_df, periods_per_year=12):
    """Per-regime summary: frequency, mean/vol/annualized return, worst month."""
    rows = []
    for name, grp in regime_df.groupby("Regime Name"):
        r = grp["Return"]
        rows.append({
            "Regime": name,
            "Months": len(r),
            "% of Time": f"{100*len(r)/len(regime_df):.0f}%",
            "Mean Monthly": f"{r.mean()*100:.2f}%",
            "Annualized Vol": f"{r.std()*np.sqrt(periods_per_year)*100:.1f}%",
            "Worst Month": f"{r.min()*100:.1f}%",
            "Best Month": f"{r.max()*100:.1f}%",
        })
    return pd.DataFrame(rows)


# ============================================================
# 9. ASSET-LIABILITY MANAGEMENT / PENSION (Course 1, Week 4)
#    Ported from edhec_risk_kit.py; the building blocks for a pension / LDI tool.
# ============================================================

def discount(t, r):
    """
    Price today of $1 paid at time t, given per-period rate r.
    Returns a DataFrame indexed by t (works if r is scalar/Series/DataFrame).
    """
    discounts = pd.DataFrame([(r + 1) ** -i for i in t])
    discounts.index = t
    return discounts


def pv(flows, r):
    """Present value of a cash-flow Series (indexed by time) at rate r."""
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis="rows").sum()


def funding_ratio(assets, liabilities, r):
    """
    Funding ratio = PV(assets) / PV(liabilities). The core pension solvency
    number: >1 means the fund can cover its promised payouts at rate r, <1 means
    a shortfall. Falls when rates fall (liabilities discounted less), which is
    exactly the interest-rate risk that liability-driven investing hedges.
    """
    return float(pv(assets, r) / pv(liabilities, r))


def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    n_coupons = round(maturity * coupons_per_year)
    coupon_amt = principal * coupon_rate / coupons_per_year
    coupon_times = np.arange(1, n_coupons + 1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows


def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
    return float(pv(cash_flows, discount_rate / coupons_per_year))


def macaulay_duration(flows, discount_rate):
    """
    Macaulay duration: the cash-flow-weighted average time to receipt. It is the
    key sensitivity number for LDI — to immunize a liability against small rate
    moves, you match the duration of your bond assets to the liability's duration.
    """
    discounted_flows = discount(flows.index, discount_rate) * flows.values.reshape(-1, 1)
    weights = discounted_flows / discounted_flows.sum()
    return float(np.average(flows.index, weights=weights.values.flatten()))


def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Weight W in the SHORT bond (and 1-W in the LONG bond) so the blended
    duration matches the target liability duration. This is the mechanical
    core of building a duration-matched liability-hedging portfolio.
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t) / (d_l - d_s)


def liability_pv_series(liabilities, rates):
    """Convenience: PV of a liability stream across a range of flat discount rates."""
    return pd.Series({r: float(pv(liabilities, r)) for r in rates})
