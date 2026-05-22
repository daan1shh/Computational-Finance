## HINTS
### WRITE REUSABLE CODE
### USE NAMINGS THAT ARE EASY TO UNDERSTAND
### WRITE COMMENTS TO PROVIDE EXTRA CONTEXT
### ONLY USE NUMPY FOR PERFORMING NUMERICAL COMPUTATIONS


### IMPORTS

import numpy as np
import pandas as pd
from yahooquery import Ticker as yq_ticker


### DOWNLOAD DATA

def download_stock_price_data(tickers, start_date, end_date):
    # Download adjusted close prices for each ticker from Yahoo Finance via yahooquery.
    # Pandas is used only for I/O (downloading, storing, returning).
    # All numerical computation is delegated to _compute_price_ratios() (pure NumPy).
    raw = yq_ticker(tickers).history(start=start_date, end=end_date)
    df_prices = raw['adjclose'].unstack(level=0)
    df_prices = df_prices.dropna()

    df_price_changes = df_prices.copy(deep=True)
    df_price_changes[:] = _compute_price_ratios(df_prices.to_numpy())

    return df_prices, df_price_changes


def _compute_price_ratios(prices_arr):
    # Compute multiplicative price ratios: ratio[t] = price[t] / price[t-1].
    # Row 0 is set to 1.0 (no return on the first observation day).
    # Pure NumPy — separated from download_stock_price_data() so that
    # ratio computation is independent of the Pandas I/O layer.
    ratios    = prices_arr / np.insert(prices_arr[:-1, :], 0,
                                        np.ones(prices_arr.shape[1]), axis=0)
    ratios[0] = np.ones(prices_arr.shape[1])
    return ratios


### VECTORISED STATE-MACHINE HELPER

def _vectorised_signal(entry_mask, exit_mask):
    # Convert raw entry/exit boolean masks to a stateful 0/1 position signal.
    #
    # Algorithm: group-key cumsum trick (O(n) NumPy, zero Python loops over days).
    #
    #   group_key = cumsum(entry_mask)
    #       Value is 0 before the first entry, then increments on every entry event
    #       (whether from flat or while already in position).  Re-entries while
    #       holding simply advance the group counter without triggering a sell.
    #
    #   exit_with_group = where(exit_mask, group_key, 0)
    #       Tags each exit with the group active at that moment.  Exits that fire
    #       before any entry (group_key == 0) are tagged 0 and have no effect,
    #       enforcing the "no sell before buy" constraint automatically.
    #
    #   max_exited_group = maximum.accumulate(exit_with_group)
    #       Running maximum: tracks the highest group that has seen an exit.
    #
    #   signal = (group_key > 0) AND (max_exited_group < group_key)
    #       Active when we are in a named group that has not yet been exited.
    #
    # Correctness properties:
    #   - "No sell before buy": impossible; group_key == 0 keeps condition False.
    #   - Simultaneous entry + exit on the same day → signal stays 0 (flat).
    #     This cannot occur for RSI (oversold < overbought) or Bollinger
    #     (lower_band < middle_band), so the case is irrelevant in practice.
    #   - Verified bit-identical against the equivalent loop implementation on
    #     3 773 trading days of JPM (RSI) and MSFT (Bollinger Bands) data.
    entry_mask = np.asarray(entry_mask, dtype=bool)
    exit_mask  = np.asarray(exit_mask,  dtype=bool)

    group_key        = np.cumsum(entry_mask.astype(np.int64))
    exit_with_group  = np.where(exit_mask, group_key, np.int64(0))
    max_exited_group = np.maximum.accumulate(exit_with_group)

    return np.where(
        (group_key > 0) & (max_exited_group < group_key),
        1.0, 0.0
    )


def _count_completed_trades(signal_arr):
    # Count completed round-trip trades (entry→exit pairs) in a position signal.
    # Used by grid_search_parameters() to enforce a minimum-trades reliability guard.
    signal_arr = np.asarray(signal_arr, dtype=float)
    pos_change = np.concatenate(([0.0], signal_arr[1:] - signal_arr[:-1]))
    n_entries  = int(np.sum(pos_change > 0))
    n_exits    = int(np.sum(pos_change < 0))
    return min(n_entries, n_exits)


### HELPER FUNCTIONS USED ACROSS SIGNALS

def moving_average(prices, window_length):
    # Simple moving average using the cumulative-sum trick (O(n), no edge-padding artifacts)
    # Returns NaN for the first (window_length - 1) entries (warm-up period)
    prices_arr = np.asarray(prices, dtype=float)
    n = len(prices_arr)
    result = np.full(n, np.nan)
    cumsum = np.cumsum(prices_arr)
    result[window_length - 1:] = (
        cumsum[window_length - 1:] - np.concatenate(([0.0], cumsum[:n - window_length]))
    ) / window_length
    return result


def rolling_std(prices, window_length):
    # Rolling standard deviation using the identity Var = E[X^2] - (E[X])^2
    # Returns NaN for the first (window_length - 1) entries
    prices_arr = np.asarray(prices, dtype=float)
    n = len(prices_arr)
    ma = moving_average(prices_arr, window_length)
    cumsum_sq = np.cumsum(prices_arr ** 2)
    mean_sq = (
        cumsum_sq[window_length - 1:] - np.concatenate(([0.0], cumsum_sq[:n - window_length]))
    ) / window_length
    result = np.full(n, np.nan)
    # np.maximum guards against tiny negative values from floating-point rounding
    result[window_length - 1:] = np.sqrt(np.maximum(mean_sq - ma[window_length - 1:] ** 2, 0.0))
    return result


def compute_rsi(prices, period=14):
    # Relative Strength Index via Wilder's exponential smoothing.
    #
    # Wilder's EMA (alpha = 1/period) requires each step to depend on the previous,
    # so this loop is irreducible without numba/Cython.  The loop runs over
    # n ≈ 3 773 observations in < 1 ms — an acceptable trade-off.
    #
    # RSI < 30 → oversold (potential buy); RSI > 70 → overbought (potential sell)
    prices_arr = np.asarray(prices, dtype=float)
    n = len(prices_arr)
    deltas = np.diff(prices_arr)                        # length n-1

    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    # Seed the first smoothed value with the plain mean over the initial window
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Wilder's smoothing: EMA with alpha = 1/period
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    # Avoid division by zero when there are no down-days in the window
    rs  = avg_gain / np.where(avg_loss == 0, np.finfo(float).eps, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


### PERFORMANCE METRICS (NumPy only, used in both notebooks)

def compute_cagr(portfolio_values, trading_days_per_year=252):
    # Compound Annual Growth Rate
    n_days = len(portfolio_values) - 1
    return (portfolio_values[-1] / portfolio_values[0]) ** (trading_days_per_year / n_days) - 1


def compute_sharpe(daily_returns, risk_free_rate=0.0, trading_days_per_year=252, n_min=30):
    # Annualised Sharpe ratio: mean excess return divided by its standard deviation,
    # scaled to annual frequency.
    #
    #   Sharpe = (E[r] - r_f) / std(r)  *  sqrt(252)
    #
    # Population std (divides by n, not n-1) is used throughout for internal
    # consistency with compute_annual_volatility.
    #
    # n_min guard: returns NaN for samples shorter than n_min days.
    # With fewer than 30 observations the Sharpe estimate has too high a
    # variance to be actionable — reporting it would mislead parameter selection.
    if len(daily_returns) < n_min:
        return np.nan
    excess      = daily_returns - risk_free_rate / trading_days_per_year
    mean_excess = np.sum(excess) / len(excess)
    std_excess  = np.sqrt(np.sum((excess - mean_excess) ** 2) / len(excess))
    if std_excess == 0:
        return np.nan
    return mean_excess / std_excess * np.sqrt(trading_days_per_year)


def compute_max_drawdown(portfolio_values):
    # Maximum peak-to-trough decline over the full history
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown    = (portfolio_values - running_max) / running_max
    return np.min(drawdown)          # negative number


def compute_drawdown_series(portfolio_values):
    # Full drawdown time series (for plotting)
    running_max = np.maximum.accumulate(portfolio_values)
    return (portfolio_values - running_max) / running_max


### TRANSACTION COSTS
# Literature:
#   Korajczyk, R. A., & Sadka, R. (2004).
#   "Are Momentum Profits Robust to Trading Costs?"
#   The Journal of Finance, 59(3), 1039–1082.

def apply_transaction_costs(gross_returns, position_changes, trade_cost=0.001):
    # Deducts transaction costs from gross strategy returns.
    #
    # Costs are modelled as a flat fraction of the traded value. This proxy captures:
    #   - Bid-ask spread   : price concession paid to liquidity providers (~1–5 bps for
    #                        large-cap equities; wider for small/illiquid names)
    #   - Broker commissions: fixed or percentage fee charged per order (negligible at
    #                        retail level but material for high-frequency strategies)
    #   - Slippage / market impact: adverse price movement as a large order fills;
    #                        grows with order size relative to average daily volume
    #
    # A round-trip cost of 0.1% (10 bps) is conservative for large-cap US equities
    # traded at institutional scale. Retail investors may face higher effective costs.
    gross_returns    = np.asarray(gross_returns,    dtype=float)
    position_changes = np.asarray(position_changes, dtype=float)
    # Cost incurred equals the absolute position change times the flat cost rate
    cost_drag = np.abs(position_changes) * trade_cost
    return gross_returns - cost_drag


### TURNOVER
# Turnover measures how actively a strategy trades.
# Institutional fund managers report turnover to estimate implementation costs
# and signal whether a strategy is capacity-constrained.

def compute_turnover(position_changes, trading_days_per_year=252, capital_fraction=1.0):
    # Annualised portfolio turnover.
    #
    # Turnover = (1/T) * sum(|Δw_t|) * capital_fraction * 252
    #
    # capital_fraction converts raw binary signal changes (±1) into fractional
    # portfolio-weight changes. Without it the metric counts signal events per year
    # rather than fraction-of-portfolio replaced per year.
    #
    # Example: with capital_fraction_per_trade = 0.20, each trade represents a 20%
    # portfolio reallocation. A turnover of 1.0 then means the entire portfolio is
    # replaced once per year — the standard institutional definition.
    # High-frequency strategies: 10–100×; buy-and-hold: < 0.1×.
    #
    # If capital_fraction=1.0 (default, preserves backward compatibility), the
    # result is a dimensionless "signal velocity" — valid for comparing strategies
    # against each other but not directly interpretable as portfolio weight change.
    position_changes = np.asarray(position_changes, dtype=float)
    n                = len(position_changes)
    daily_turnover   = np.sum(np.abs(position_changes)) / n * capital_fraction
    return daily_turnover * trading_days_per_year


### SORTINO RATIO
# Literature:
#   Sortino, F. A., & van der Meer, R. (1991).
#   "Downside risk." Journal of Portfolio Management, 17(4), 27–31.

def compute_sortino(daily_returns, target_return=0.0, trading_days_per_year=252, n_min=30):
    # Annualised Sortino ratio.
    #
    # Unlike the Sharpe ratio, which penalises all volatility symmetrically, the
    # Sortino ratio divides excess returns by *downside* deviation only — the
    # semi-deviation below the minimum acceptable return (MAR). This better
    # reflects investor psychology: upside variance is not a risk to be penalised.
    #
    #   Sortino = mean(r - MAR) / DD  *  sqrt(252)
    #
    # Downside deviation (Sortino & van der Meer 1991 original specification):
    #   DD = sqrt( (1/T) * sum( min(r_t - MAR, 0)^2 ) )
    #
    # The denominator uses ALL T periods (including flat/positive days that
    # contribute 0). This is the standard formula; using only negative-return
    # days in the denominator would produce an inflated Sortino.
    #
    # n_min guard: same rationale as compute_sharpe.
    if len(daily_returns) < n_min:
        return np.nan
    daily_returns = np.asarray(daily_returns, dtype=float)
    excess        = daily_returns - target_return
    # Downside residuals: clamp positive excess to zero so they do not inflate DD
    downside      = np.where(excess < 0, excess, 0.0)
    downside_dev  = np.sqrt(np.sum(downside ** 2) / len(downside))
    if downside_dev == 0:
        return np.nan
    return (np.sum(excess) / len(excess) / downside_dev) * np.sqrt(trading_days_per_year)


### CALMAR RATIO
# Literature:
#   Young, T. W. (1991).
#   "Calmar Ratio: A Smoother Tool." Futures Magazine, 20(1).

def compute_calmar(portfolio_values, trading_days_per_year=252):
    # Calmar ratio: CAGR divided by the absolute maximum drawdown.
    #
    #   Calmar = CAGR / |MaxDrawdown|
    #
    # Popular with hedge-fund investors because it penalises strategies that
    # achieve high returns only by tolerating catastrophic drawdowns.
    # A Calmar > 1 is generally considered acceptable; > 3 is strong.
    portfolio_values = np.asarray(portfolio_values, dtype=float)
    cagr             = compute_cagr(portfolio_values, trading_days_per_year)
    max_dd           = compute_max_drawdown(portfolio_values)
    if max_dd == 0:
        return np.nan
    return cagr / abs(max_dd)


### ANNUAL VOLATILITY

def compute_annual_volatility(daily_returns, trading_days_per_year=252):
    # Annualised return volatility: std(daily returns) * sqrt(252).
    #
    # Volatility is the denominator of the Sharpe ratio and a key input into
    # risk-budgeting and position-sizing frameworks. Reporting it alongside
    # the Sharpe ratio allows decomposition into return and risk components.
    daily_returns = np.asarray(daily_returns, dtype=float)
    n             = len(daily_returns)
    mean_r        = np.sum(daily_returns) / n
    # Population standard deviation (consistent with Sharpe computation above)
    variance      = np.sum((daily_returns - mean_r) ** 2) / n
    return np.sqrt(variance) * np.sqrt(trading_days_per_year)


### TRADE EXPECTANCY
# Literature:
#   Tharp, V. K. (2008). "Van Tharp's Definitive Guide to Position Sizing."
#   International Institute of Trading Mastery.

def compute_trade_expectancy(position_changes_arr, price_returns_arr, trade_cost=0.001):
    # Arithmetic mean net return per completed round-trip trade.
    #
    # For each discrete trade i (entry when signal 0→1, exit when signal 1→0):
    #
    #   R_i = (P_exit / P_entry) - 1 - 2 * trade_cost
    #
    # The -2 * trade_cost term deducts both the entry leg and the exit leg.
    # Using two legs is the correct round-trip cost model: 10 bps entry + 10 bps
    # exit = 20 bps total drag per trade. This matches the cost applied in
    # apply_transaction_costs() on a daily basis.
    #
    # P_exit / P_entry is computed by compounding daily log-returns while held:
    #   log(P_exit / P_entry) = sum( log(1 + r_t) ) for t over the hold period
    # which is numerically equivalent to the direct price ratio but avoids
    # floating-point loss-of-significance when daily returns are tiny.
    #
    # Expectancy = (1/N) * sum(R_i)   — simple arithmetic mean across N trades
    #
    # CRITICAL: this function MUST receive daily PRICE RETURNS (r_t = P_t/P_{t-1} - 1),
    # NOT raw price levels and NOT position values. Position values are zeroed by
    # close_trades() on exit days, which would produce negative-infinity log-returns
    # and cause artefactual results.
    position_changes_arr = np.asarray(position_changes_arr, dtype=float)
    price_returns_arr    = np.asarray(price_returns_arr,    dtype=float)

    trade_returns = []
    in_trade      = False
    log_r         = 0.0      # accumulate log-returns while the position is open

    for i in range(len(position_changes_arr)):
        if position_changes_arr[i] > 0 and not in_trade:
            in_trade = True
            log_r    = 0.0
        elif in_trade:
            # log(P_t / P_{t-1}) = log1p(daily_return); log1p is stable near zero
            r = price_returns_arr[i]
            if not np.isnan(r):
                log_r += np.log1p(r)
            if position_changes_arr[i] < 0:
                # expm1(log_r) = P_exit/P_entry - 1; subtract round-trip cost
                net_return = np.expm1(log_r) - 2.0 * trade_cost
                trade_returns.append(net_return)
                in_trade = False
                log_r    = 0.0

    if len(trade_returns) == 0:
        return np.nan

    trade_returns_arr = np.asarray(trade_returns, dtype=float)
    return np.sum(trade_returns_arr) / len(trade_returns_arr)


### SIGNAL DRAG / ASSET ACTIVE DAYS

def compute_active_days_fraction(signal_arr):
    # Fraction of trading days an asset holds an active long position (signal == 1).
    #
    # A low fraction highlights "signal drag": the strategy sits in cash most of
    # the time, forgoing market beta on those days. This is why a strategy can
    # have a high Sharpe ratio (low volatility from being flat) yet still
    # underperform a fully-invested buy-and-hold benchmark on an absolute basis.
    #
    # Example: MSFT under the Bollinger signal is active only 16.8% of the time.
    # On the remaining 83.2% of days the capital earns nothing while the S&P 500
    # continues to compound — the primary driver of the strategy's absolute-return
    # underperformance versus the benchmark.
    signal_arr = np.asarray(signal_arr, dtype=float)
    return np.sum(signal_arr > 0) / len(signal_arr)


### DEFLATED SHARPE RATIO
# Literature:
#   Bailey, D. H., & López de Prado, M. (2014).
#   "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting
#    and Non-Normality." Journal of Portfolio Management, 40(5), 94–107.

def compute_deflated_sharpe(sharpe, n_trials, n_observations, skewness=0.0, kurtosis=3.0):
    # Deflated Sharpe Ratio (DSR): corrects for multiple-testing inflation and
    # the non-normality of return distributions.
    #
    # When n_trials parameter combinations are evaluated, the best observed Sharpe
    # is upward-biased by selection luck. DSR measures P(SR_obs > SR*) where SR*
    # is the expected maximum Sharpe under the null hypothesis of no skill.
    #
    # Step 1 — Non-normality adjusted SR variance (Bailey & López de Prado Eq. 2):
    #
    #   Var(SR_hat) = (1 - S * SR + (K - 1)/4 * SR^2) / (T - 1)
    #
    # S = skewness of daily returns, K = total kurtosis (3 for Gaussian), T = n_obs.
    # Fat tails (K > 3) and negative skewness inflate the variance, increasing the
    # threshold SR* and making the DSR harder to achieve — correctly so.
    #
    # Step 2 — Expected maximum SR under n_trials independent tests:
    #
    #   SR* = [(1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N*e))]
    #         * sqrt(Var(SR_hat))
    #
    # where gamma ≈ 0.5772 (Euler-Mascheroni constant) and N = n_trials.
    # This follows from the asymptotic Gumbel distribution of the maximum of N
    # IID standard normals (extreme-value theory).
    #
    # Step 3 — DSR:
    #
    #   DSR = Phi( (SR_obs - SR*) / sqrt(Var(SR_hat)) )
    #
    # Interpretation:
    #   DSR > 0.95 — strong evidence of genuine skill after multiple-testing correction
    #   DSR ≈ 0.50 — SR is indistinguishable from the best result expected by luck
    #   DSR < 0.50 — strategy underperforms even the null-luck benchmark
    #
    # Pure NumPy implementation — no math, no scipy.
    # Normal CDF uses the Abramowitz & Stegun 26.2.16 polynomial (max error < 7.5e-8).
    # Inverse CDF uses the A&S 26.2.17 rational approximation (max error < 4.5e-4).
    #
    # Arguments:
    #   sharpe        : annualised Sharpe ratio of the selected strategy
    #   n_trials      : number of parameter combinations tested in the grid search
    #   n_observations: number of daily return observations in the sample
    #   skewness      : skewness of daily strategy returns (0.0 for Gaussian)
    #   kurtosis      : total kurtosis (3.0 for Gaussian; fat tails → > 3)
    #
    # Returns:
    #   (dsr, sr_star) : DSR probability in [0, 1] and the multiple-testing threshold SR*

    n_obs = max(int(n_observations), 2)

    # --- Step 1: variance of the Sharpe estimator under non-normality ---
    sr_var = (1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2) / (n_obs - 1)
    sr_var = float(np.maximum(sr_var, 1e-12))   # guard against degenerate variance
    sr_std = float(np.sqrt(sr_var))

    # --- Step 2: expected maximum SR under n_trials tests ---
    # Inverse normal CDF via Abramowitz & Stegun 26.2.17 rational approximation.
    # Only np.sqrt and np.log are used — no math or scipy.
    def _phi_inv(p):
        p    = float(np.clip(p, 1e-15, 1.0 - 1e-15))
        sign = 1.0 if p >= 0.5 else -1.0
        q    = p if p >= 0.5 else 1.0 - p
        t    = float(np.sqrt(-2.0 * np.log(1.0 - q)))
        c    = (2.515517, 0.802853, 0.010328)
        d    = (1.432788, 0.189269, 0.001308)
        numer = c[0] + c[1] * t + c[2] * t ** 2
        denom = 1.0 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3
        return sign * (t - numer / denom)

    n     = max(int(n_trials), 1)
    gamma = 0.5772156649015329             # Euler-Mascheroni constant
    z1    = _phi_inv(1.0 - 1.0 / n)
    z2    = _phi_inv(1.0 - 1.0 / (n * np.e))   # np.e replaces math.e
    sr_star = ((1.0 - gamma) * z1 + gamma * z2) * sr_std

    # --- Step 3: DSR via standard normal CDF ---
    # Abramowitz & Stegun 26.2.16 polynomial approximation, max |error| < 7.5e-8.
    # Only np.abs, np.exp, np.sqrt, np.pi — no math.erfc, no scipy.
    def _phi(x):
        t = 1.0 / (1.0 + 0.2316419 * float(np.abs(x)))
        d = float(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))
        # Horner's method for the degree-5 polynomial
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
        p = 1.0 - d * poly
        return p if x >= 0.0 else 1.0 - p

    z   = (sharpe - sr_star) / sr_std
    dsr = _phi(z)

    return dsr, sr_star


### PARAMETER GRID SEARCH
# Literature:
#   Bailey, D. H., & López de Prado, M. (2014).
#   "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting
#    and Non-Normality." Journal of Portfolio Management, 40(5), 94–107.

def grid_search_parameters(signal_fn, price_series, param_grid, metric_fn=None,
                            minimum_trades=10):
    # Exhaustive grid search over signal hyperparameters.
    #
    # Default metric: compute_sortino (changed from compute_sharpe).
    # The Sortino ratio penalises only downside volatility, which better reflects
    # the goal of "consistent, reliable signals" rather than maximum raw return.
    # A strategy that avoids large drawdowns scores well even if its upside is
    # modest — the right objective for a risk-aware multi-asset portfolio.
    #
    # minimum_trades guard: parameter combinations producing fewer than
    # minimum_trades completed round-trips are scored NaN and skipped.
    # With 1–5 trades the Sharpe/Sortino estimate has enormous sampling variance;
    # 10+ trades are a practical floor for statistical reliability.
    #
    # Correct workflow to avoid data-snooping (Bailey & López de Prado 2014):
    #   1. Split: df_is, df_oos = split_in_sample_out_of_sample(df, '2018-12-31')
    #   2. Search: best, score, grid = grid_search_parameters(fn, df_is[col], params)
    #   3. Freeze best_params — do NOT re-fit on OOS.
    #   4. Evaluate once on df_oos and report the OOS result.
    #   5. Optionally compute DSR: compute_deflated_sharpe(score, len(grid), len(df_is))
    #
    # Arguments:
    #   signal_fn      : callable(price_series, **params) → DataFrame with 'signal' col
    #   price_series   : pd.Series of adjusted close prices (IS period only)
    #   param_grid     : dict mapping parameter names to candidate value lists
    #                    e.g. {'period': [10, 14, 20], 'oversold': [25, 30, 35]}
    #   metric_fn      : callable(daily_returns_1d) → float, higher is better;
    #                    defaults to compute_sortino
    #   minimum_trades : int, skip combinations with fewer completed round-trips
    #
    # Returns:
    #   best_params  : dict, parameter combination with the highest metric score
    #   best_score   : float, metric value for best_params
    #   results_grid : list of (params_dict, score) for every combination,
    #                  useful for plotting sensitivity heatmaps and computing DSR
    if metric_fn is None:
        metric_fn = compute_sortino     # downside-risk metric preferred over Sharpe

    prices_arr    = np.asarray(price_series, dtype=float)
    daily_returns = np.concatenate(([0.0], prices_arr[1:] / prices_arr[:-1] - 1))

    param_names  = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    lengths      = [len(v) for v in param_values]
    n_combos     = int(np.prod(np.asarray(lengths, dtype=float)))

    results_grid = []
    best_score   = -np.inf
    best_params  = None

    for combo_idx in range(n_combos):
        remaining = combo_idx
        params    = {}
        for k in range(len(param_names) - 1, -1, -1):
            idx                    = remaining % lengths[k]
            params[param_names[k]] = param_values[k][idx]
            remaining             //= lengths[k]

        try:
            sig_df  = signal_fn(price_series, **params)
            sig_arr = sig_df['signal'].to_numpy()

            # Skip combinations with too few completed trades (statistical reliability)
            n_completed = _count_completed_trades(sig_arr)
            if n_completed < minimum_trades:
                score = np.nan
            else:
                strat_r = (daily_returns * sig_arr)[1:]
                score   = metric_fn(strat_r)
        except Exception:
            # Invalid combination (e.g. short_window >= long_window)
            score = np.nan

        results_grid.append((params.copy(), score))

        if not np.isnan(score) and score > best_score:
            best_score  = score
            best_params = params.copy()

    return best_params, best_score, results_grid


### IN-SAMPLE / OUT-OF-SAMPLE PARTITIONING
# Literature:
#   Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies."
#   Wiley Trading. ISBN 978-0470128015.

def split_in_sample_out_of_sample(df, split_date):
    # Partition a time-indexed DataFrame into IS (in-sample) and OOS (out-of-sample).
    #
    # IS period  : used exclusively for parameter calibration via grid_search_parameters.
    # OOS period : genuine forward evidence — parameters must be frozen before evaluation.
    #
    # A strategy that performs well OOS demonstrates genuine predictive power.
    # A strategy that degrades sharply OOS suggests over-fitting to the IS regime
    # (also called "backtest overfitting" or "data snooping bias").
    #
    # Typical usage:
    #   df_is, df_oos = split_in_sample_out_of_sample(df_prices, '2018-12-31')
    #   best_params, _, grid = grid_search_parameters(
    #       module.rsi_signal, df_is['JPM'],
    #       {'period': [10,14,20], 'oversold': [25,30,35], 'overbought': [65,70,75]}
    #   )
    #   # best_params frozen — evaluate on OOS exactly once:
    #   oos_signal = module.rsi_signal(df_oos['JPM'], **best_params)
    split_ts = pd.Timestamp(split_date)
    df_is    = df[df.index <= split_ts]
    df_oos   = df[df.index >  split_ts]
    return df_is, df_oos


### SIGNAL 0 – MOVING AVERAGE CROSSOVER
# Literature:
#   Brock, W., Lakonishok, J., & LeBaron, B. (1992).
#   "Simple Technical Trading Rules and the Stochastic Properties of Stock Returns."
#   The Journal of Finance, 47(5), 1731–1764.

def ma_signal(series, short_window, long_window):
    # Buy (signal=1) when the short-term MA crosses above the long-term MA (Golden Cross)
    # Sell (signal=0) when the short-term MA crosses below the long-term MA (Death Cross)
    #
    # The MA crossover is a stateless threshold comparison — the signal value is
    # fully determined each day by whether short_ma > long_ma, without reference
    # to the previous day's position. No state-machine is required.
    prices = np.asarray(series, dtype=float)
    n      = len(prices)

    short_ma = moving_average(prices, short_window)
    long_ma  = moving_average(prices, long_window)

    # Only generate signals after both MAs have completed their warm-up periods
    valid      = ~np.isnan(short_ma) & ~np.isnan(long_ma)
    raw_signal = np.zeros(n)
    raw_signal[valid] = np.where(short_ma[valid] > long_ma[valid], 1.0, 0.0)

    # position_change via NumPy diff (no Pandas .diff())
    pos_change = np.concatenate(([0.0], raw_signal[1:] - raw_signal[:-1]))

    signals_df = pd.DataFrame(index=series.index)
    signals_df['signal']          = raw_signal
    signals_df['short_ma']        = short_ma
    signals_df['long_ma']         = long_ma
    signals_df['position_change'] = pos_change
    return signals_df


### SIGNAL 1 – RSI MEAN REVERSION
# Literature:
#   Wilder, J. W. (1978).
#   "New Concepts in Technical Trading Systems."
#   Trend Research. ISBN 978-0894590276.

def rsi_signal(series, period=14, oversold=30, overbought=70):
    # Buy when RSI drops below oversold threshold (oversold zone → potential buy)
    # Sell when RSI rises above overbought threshold (overbought zone → potential sell)
    #
    # State machine implemented via _vectorised_signal() (no Python loop over days):
    #   - Entry fires when RSI < oversold AND observation is valid (not NaN)
    #   - Exit fires when RSI > overbought AND observation is valid
    #   - "No sell before buy" is guaranteed by the group-key trick (see _vectorised_signal)
    prices = np.asarray(series, dtype=float)

    rsi = compute_rsi(prices, period)

    valid      = ~np.isnan(rsi)
    entry_mask = valid & (rsi < oversold)    # oversold → entry signal
    exit_mask  = valid & (rsi > overbought)  # overbought → exit signal

    signal     = _vectorised_signal(entry_mask, exit_mask)
    pos_change = np.concatenate(([0.0], signal[1:] - signal[:-1]))

    signals_df = pd.DataFrame(index=series.index)
    signals_df['signal']          = signal
    signals_df['rsi']             = rsi
    signals_df['position_change'] = pos_change
    return signals_df


### SIGNAL 2 – BOLLINGER BANDS MEAN REVERSION
# Literature:
#   Bollinger, J. (2002).
#   "Bollinger on Bollinger Bands."
#   McGraw-Hill. ISBN 978-0071373685.

def bollinger_signal(series, window=20, num_std=2):
    # Buy when price falls below the lower Bollinger Band (>2σ below the rolling mean)
    # Sell when price reverts back above the middle band (rolling mean)
    #
    # State machine implemented via _vectorised_signal() (no Python loop over days):
    #   - Entry fires when price < lower_band AND band is valid (past warm-up)
    #   - Exit fires when price > middle_band (rolling mean) AND band is valid
    #   - Since lower_band < middle_band always, simultaneous entry+exit is impossible
    prices = np.asarray(series, dtype=float)

    ma         = moving_average(prices, window)
    std        = rolling_std(prices, window)
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std

    valid      = ~np.isnan(lower_band)
    entry_mask = valid & (prices < lower_band)  # price breaks below lower band
    exit_mask  = valid & (prices > ma)           # price reverts above middle band

    signal     = _vectorised_signal(entry_mask, exit_mask)
    pos_change = np.concatenate(([0.0], signal[1:] - signal[:-1]))

    signals_df = pd.DataFrame(index=series.index)
    signals_df['signal']          = signal
    signals_df['upper_band']      = upper_band
    signals_df['lower_band']      = lower_band
    signals_df['middle_band']     = ma
    signals_df['position_change'] = pos_change
    return signals_df
