## HINTS
### WRITE REUSABLE CODE
### USE NAMINGS THAT ARE EASY TO UNDERSTAND
### WRITE COMMENTS TO PROVIDE EXTRA CONTEXT
### ONLY USE NUMPY FOR PERFORMING NUMERICAL COMPUTATIONS

### IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahooquery import Ticker as yq_ticker


### DOWNLOAD DATA

def download_stock_price_data(tickers, start_date, end_date):
    # Download adjusted close prices and compute daily price ratios.
    raw = yq_ticker(tickers).history(start=start_date, end=end_date)
    df_prices = raw['adjclose'].unstack(level=0)
    df_prices = df_prices.dropna()

    df_price_changes = df_prices.copy(deep=True)
    df_price_changes[:] = _compute_price_ratios(df_prices.to_numpy())

    return df_prices, df_price_changes


def _compute_price_ratios(prices_arr):
    # Price ratio at each row: price[t] / price[t-1]. Row 0 set to 1.0.
    ratios    = prices_arr / np.insert(prices_arr[:-1, :], 0,
                                        np.ones(prices_arr.shape[1]), axis=0)
    ratios[0] = np.ones(prices_arr.shape[1])
    return ratios


### VECTORISED STATE-MACHINE HELPER

def _vectorised_signal(entry_mask, exit_mask):
    # Convert entry/exit boolean masks to a stateful 0/1 signal.
    # Uses the group-key cumsum trick (O(n) NumPy, no Python loop over days).
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
    # Count completed round-trip trades (entry→exit pairs).
    signal_arr = np.asarray(signal_arr, dtype=float)
    pos_change = np.concatenate(([0.0], signal_arr[1:] - signal_arr[:-1]))
    n_entries  = int(np.sum(pos_change > 0))
    n_exits    = int(np.sum(pos_change < 0))
    return min(n_entries, n_exits)


### HELPER FUNCTIONS USED ACROSS SIGNALS

def moving_average(prices, window_length):
    # Simple MA via cumsum trick. NaN for the first (window_length-1) entries.
    prices_arr = np.asarray(prices, dtype=float)
    n = len(prices_arr)
    result = np.full(n, np.nan)
    cumsum = np.cumsum(prices_arr)
    result[window_length - 1:] = (
        cumsum[window_length - 1:] - np.concatenate(([0.0], cumsum[:n - window_length]))
    ) / window_length
    return result


def rolling_std(prices, window_length):
    # Rolling std using Var = E[X²] - (E[X])². NaN for first (window_length-1) entries.
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
    # RSI via Wilder's EMA (alpha = 1/period). Loop is irreducible without numba.
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
    # Annualised Sharpe: (mean excess return / std) * sqrt(252).
    # Returns NaN for samples shorter than n_min days.
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
    # Deducts flat per-trade cost from gross daily returns based on absolute position changes.
    gross_returns    = np.asarray(gross_returns,    dtype=float)
    position_changes = np.asarray(position_changes, dtype=float)
    # Cost incurred equals the absolute position change times the flat cost rate
    cost_drag = np.abs(position_changes) * trade_cost
    return gross_returns - cost_drag


### TURNOVER

def compute_turnover(position_changes, trading_days_per_year=252, capital_fraction=1.0):
    # Annualised turnover: (1/T) * sum(|Δw|) * capital_fraction * 252.
    position_changes = np.asarray(position_changes, dtype=float)
    n                = len(position_changes)
    daily_turnover   = np.sum(np.abs(position_changes)) / n * capital_fraction
    return daily_turnover * trading_days_per_year


### SORTINO RATIO
# Literature:
#   Sortino, F. A., & van der Meer, R. (1991).
#   "Downside risk." Journal of Portfolio Management, 17(4), 27–31.

def compute_sortino(daily_returns, target_return=0.0, trading_days_per_year=252, n_min=30):
    # Sortino ratio: mean(r - MAR) / downside_deviation * sqrt(252).
    # Downside deviation uses all T periods (Sortino & van der Meer 1991).
    # Returns NaN for samples shorter than n_min days.
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
    # Calmar ratio: CAGR / |max drawdown|.
    portfolio_values = np.asarray(portfolio_values, dtype=float)
    cagr             = compute_cagr(portfolio_values, trading_days_per_year)
    max_dd           = compute_max_drawdown(portfolio_values)
    if max_dd == 0:
        return np.nan
    return cagr / abs(max_dd)


### ANNUAL VOLATILITY

def compute_annual_volatility(daily_returns, trading_days_per_year=252):
    # Annualised volatility: std(daily returns) * sqrt(252).
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

def _collect_trade_returns(position_changes_arr, price_returns_arr, trade_cost=0.001):
    # Return a list of net log-returns for each completed round-trip trade.
    # Shared by compute_trade_expectancy and any caller that needs the full distribution
    # (e.g. to compute cross-ticker pooled mean, median, or win-rate on net returns).
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

    return trade_returns


def compute_trade_expectancy(position_changes_arr, price_returns_arr, trade_cost=0.001):
    # Mean net return per completed round-trip trade.
    # Uses log-returns for numerical stability; deducts 2 * trade_cost for the round trip.
    # Receives daily PRICE RETURNS — not position values (zeroed on exit by close_trades).
    trade_returns = _collect_trade_returns(position_changes_arr, price_returns_arr, trade_cost)
    if len(trade_returns) == 0:
        return np.nan
    trade_returns_arr = np.asarray(trade_returns, dtype=float)
    return float(np.sum(trade_returns_arr) / len(trade_returns_arr))


### SIGNAL DRAG / ASSET ACTIVE DAYS

def compute_active_days_fraction(signal_arr):
    # Fraction of trading days the strategy holds a long position (signal == 1).
    signal_arr = np.asarray(signal_arr, dtype=float)
    return np.sum(signal_arr > 0) / len(signal_arr)


### DEFLATED SHARPE RATIO
# Literature:
#   Bailey, D. H., & López de Prado, M. (2014).
#   "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting
#    and Non-Normality." Journal of Portfolio Management, 40(5), 94–107.

def compute_deflated_sharpe(sharpe, n_trials, n_observations, skewness=0.0, kurtosis=3.0):
    # Deflated Sharpe Ratio (Bailey & López de Prado 2014).
    # Corrects observed SR for multiple-testing inflation across n_trials parameter combinations.
    # Uses A&S polynomial approximations for normal CDF/inverse-CDF (no scipy needed).

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

### BASKET SORTINO — multi-ETF scoring helper

def basket_sortino(signal_fn, df_basket, **params):
    # Mean IS Sortino across all ETFs in basket with 1-day signal lag.
    # Returns NaN if any ETF produces zero completed round-trip trades.
    scores = []
    for col in df_basket.columns:
        px  = df_basket[col].to_numpy(dtype=float)
        dr  = np.concatenate(([0.0], px[1:] / px[:-1] - 1))
        try:
            sig = signal_fn(df_basket[col], **params)
            arr = sig['signal'].to_numpy(dtype=float)
            pc  = sig['position_change'].to_numpy(dtype=float)
            if min(int(np.sum(pc > 0)), int(np.sum(pc < 0))) < 1:
                return float('nan')
            strat = dr[1:] * arr[:-1]   # signal[t-1] * return[t]  — 1-day lag
            s = compute_sortino(strat)
            scores.append(s if s == s else float('nan'))
        except Exception:
            scores.append(float('nan'))
    valid = [s for s in scores if s == s]
    return float(np.mean(valid)) if valid else float('nan')


### PARAMETER GRID SEARCH
# Literature:
#   Bailey, D. H., & López de Prado, M. (2014).
#   "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting
#    and Non-Normality." Journal of Portfolio Management, 40(5), 94–107.

def grid_search_parameters(signal_fn, price_series, param_grid, metric_fn=None,
                            minimum_trades=10):
    # Exhaustive grid search over signal hyperparameters, scored by metric_fn (default: Sortino).
    # Skips combinations with fewer than minimum_trades completed round-trips.
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

            n_completed = _count_completed_trades(sig_arr)
            if n_completed < minimum_trades:
                score = np.nan
            else:
                position = np.concatenate(([0.0], sig_arr[:-1]))
                strat_r  = (daily_returns * position)[1:]
                score    = metric_fn(strat_r)
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
    # Split a time-indexed DataFrame into IS (≤ split_date) and OOS (> split_date).
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
    # Buy when short MA > long MA (Golden Cross); sell on the reverse.
    prices = np.asarray(series, dtype=float)
    n      = len(prices)

    short_ma = moving_average(prices, short_window)
    long_ma  = moving_average(prices, long_window)

    # only generate signals after both MAs have warmed up
    valid      = ~np.isnan(short_ma) & ~np.isnan(long_ma)
    raw_signal = np.zeros(n)
    raw_signal[valid] = np.where(short_ma[valid] > long_ma[valid], 1.0, 0.0)

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
    # Buy when RSI < oversold; sell when RSI > overbought.
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
    # Buy when price falls below lower Bollinger Band; sell when it reverts above the mean.
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


def exponential_moving_average(prices, span):
    # EMA with alpha = 2/(span+1), seeded with SMA of the first span observations.
    prices_arr = np.asarray(prices, dtype=float)
    n     = len(prices_arr)
    alpha = 2.0 / (span + 1)
    ema   = np.full(n, np.nan)
    if n < span:
        return ema
    ema[span - 1] = np.mean(prices_arr[:span])
    for i in range(span, n):
        ema[i] = alpha * prices_arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def macd_signal(series, fast_span=12, slow_span=26, signal_span=9):
    # MACD: buy when MACD line crosses above the signal line, sell when it crosses below.
    prices = np.asarray(series, dtype=float)

    ema_fast   = exponential_moving_average(prices, fast_span)
    ema_slow   = exponential_moving_average(prices, slow_span)
    macd_line  = ema_fast - ema_slow

    # signal line is EMA of macd_line — seed from first valid macd value
    sig_line   = np.full(len(prices), np.nan)
    first_valid = slow_span - 1
    if len(prices) > first_valid + signal_span:
        macd_valid = macd_line[first_valid:]
        sig_valid  = exponential_moving_average(macd_valid, signal_span)
        sig_line[first_valid:] = sig_valid

    histogram  = macd_line - sig_line

    valid      = ~np.isnan(histogram)
    entry_mask = valid & (macd_line > sig_line)
    exit_mask  = valid & (macd_line < sig_line)

    signal     = _vectorised_signal(entry_mask, exit_mask)
    pos_change = np.concatenate(([0.0], signal[1:] - signal[:-1]))

    signals_df = pd.DataFrame(index=series.index)
    signals_df['signal']          = signal
    signals_df['macd_line']       = macd_line
    signals_df['signal_line']     = sig_line
    signals_df['histogram']       = histogram
    signals_df['position_change'] = pos_change
    return signals_df


def zscore_signal(series, window=20, entry_threshold=2.0, exit_threshold=0.0):
    # Buy when z-score < -entry_threshold (oversold); sell when z-score > exit_threshold.
    prices = np.asarray(series, dtype=float)

    ma     = moving_average(prices, window)
    std    = rolling_std(prices, window)

    with np.errstate(invalid='ignore', divide='ignore'):
        zscore = np.where(std > 0, (prices - ma) / std, np.nan)

    valid      = ~np.isnan(zscore)
    entry_mask = valid & (zscore < -entry_threshold)
    exit_mask  = valid & (zscore > exit_threshold)

    signal     = _vectorised_signal(entry_mask, exit_mask)
    pos_change = np.concatenate(([0.0], signal[1:] - signal[:-1]))

    signals_df = pd.DataFrame(index=series.index)
    signals_df['signal']          = signal
    signals_df['zscore']          = zscore
    signals_df['position_change'] = pos_change
    return signals_df


def donchian_signal(series, window=55, entry_window=None, exit_window=None):
    # Donchian Channel Breakout: buy on new N-day high, exit on M-day low.
    if entry_window is None:
        entry_window = window
    if exit_window is None:
        exit_window = window

    prices = np.asarray(series, dtype=float)
    n      = len(prices)

    from numpy.lib.stride_tricks import sliding_window_view

    # Entry: price > highest high over last entry_window days (excluding today)
    entry_high = np.full(n, np.nan)
    if n > entry_window:
        wins = sliding_window_view(prices[:-1], entry_window)   # shape (n-entry_window, entry_window)
        entry_high[entry_window:] = np.max(wins, axis=1)

    # Exit: price < lowest low over last exit_window days (excluding today)
    exit_low = np.full(n, np.nan)
    if n > exit_window:
        wins = sliding_window_view(prices[:-1], exit_window)
        exit_low[exit_window:] = np.min(wins, axis=1)

    valid      = ~np.isnan(entry_high) & ~np.isnan(exit_low)
    entry_mask = valid & (prices > entry_high)
    exit_mask  = valid & (prices < exit_low)

    signal     = _vectorised_signal(entry_mask, exit_mask)
    pos_change = np.concatenate(([0.0], signal[1:] - signal[:-1]))

    signals_df = pd.DataFrame(index=series.index)
    signals_df['signal']          = signal
    signals_df['entry_high']      = entry_high
    signals_df['exit_low']        = exit_low
    signals_df['position_change'] = pos_change
    return signals_df


def stochastic_signal(series, k_window=14, d_window=3, oversold=20, overbought=80):
    # Stochastic oscillator: buy when %K < oversold, sell when %K > overbought.
    prices = np.asarray(series, dtype=float)
    n      = len(prices)

    from numpy.lib.stride_tricks import sliding_window_view
    windows      = sliding_window_view(prices, k_window)
    highest_high = np.full(n, np.nan)
    lowest_low   = np.full(n, np.nan)
    highest_high[k_window - 1:] = np.max(windows, axis=1)
    lowest_low[k_window - 1:]   = np.min(windows, axis=1)

    denom = highest_high - lowest_low
    with np.errstate(invalid='ignore', divide='ignore'):
        pct_k = np.where(denom > 0, (prices - lowest_low) / denom * 100, 50.0)
    pct_k[:k_window - 1] = np.nan

    pct_d = moving_average(pct_k, d_window)

    valid      = ~np.isnan(pct_k)
    entry_mask = valid & (pct_k < oversold)
    exit_mask  = valid & (pct_k > overbought)

    signal     = _vectorised_signal(entry_mask, exit_mask)
    pos_change = np.concatenate(([0.0], signal[1:] - signal[:-1]))

    signals_df = pd.DataFrame(index=series.index)
    signals_df['signal']          = signal
    signals_df['pct_k']           = pct_k
    signals_df['pct_d']           = pct_d
    signals_df['position_change'] = pos_change
    return signals_df


### VISUALISATION HELPER

def draw_heatmap(ax, data, row_labels, col_labels, row_title, col_title,
                 title, star_row, star_col, colorbar_label='Sortino'):
    # Colour-coded heatmap with annotated cell values; ★ marks the chosen parameter.
    # colorbar_label: override the default 'Sortino' label on the colour bar.
    vmin = float(np.nanmin(data)) if not np.all(np.isnan(data)) else -1
    vmax = float(np.nanmax(data)) if not np.all(np.isnan(data)) else  1
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel(col_title, fontsize=9)
    ax.set_ylabel(row_title, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    mid = (vmin + vmax) / 2
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if not np.isnan(data[r, c]):
                marker = ' ★' if (r == star_row and c == star_col) else ''
                tc = 'black' if data[r, c] > mid else 'white'
                ax.text(c, r, f'{data[r,c]:.2f}{marker}',
                        ha='center', va='center', fontsize=7, color=tc,
                        fontweight='bold' if marker else 'normal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
