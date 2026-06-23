### IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahooquery import Ticker as yq_ticker


### DOWNLOAD DATA

def download_stock_price_data(tickers, start_date, end_date):
    raw = yq_ticker(tickers).history(start=start_date, end=end_date)
    df_prices = raw['adjclose'].unstack(level=0)
    df_prices = df_prices.dropna()

    df_price_changes = df_prices.copy(deep=True)
    df_price_changes[:] = _compute_price_ratios(df_prices.to_numpy())

    return df_prices, df_price_changes


def _compute_price_ratios(prices_arr):
    # ratio[t] = price[t] / price[t-1]; row 0 set to 1.
    ratios    = prices_arr / np.insert(prices_arr[:-1, :], 0,
                                        np.ones(prices_arr.shape[1]), axis=0)
    ratios[0] = np.ones(prices_arr.shape[1])
    return ratios


def make_price_changes(df_prices):
    arr    = df_prices.to_numpy()
    ratios = np.ones_like(arr)
    ratios[1:] = arr[1:] / arr[:-1]
    return pd.DataFrame(ratios, index=df_prices.index, columns=df_prices.columns)


### SIGNALS — VECTORISED HELPERS

def _vectorised_signal(entry_mask, exit_mask):
    # cumsum group-key trick: avoids a Python loop for stateful 0/1 signal.
    entry_mask = np.asarray(entry_mask, dtype=bool)
    exit_mask  = np.asarray(exit_mask,  dtype=bool)

    group_key        = np.cumsum(entry_mask.astype(np.int64))
    exit_with_group  = np.where(exit_mask, group_key, np.int64(0))
    max_exited_group = np.maximum.accumulate(exit_with_group)

    return np.where(
        (group_key > 0) & (max_exited_group < group_key),
        1.0, 0.0
    )


### HELPER FUNCTIONS

def moving_average(prices, window_length):
    # cumsum trick; NaN for the first (window_length-1) entries.
    prices_arr = np.asarray(prices, dtype=float)
    n = len(prices_arr)
    result = np.full(n, np.nan)
    cumsum = np.cumsum(prices_arr)
    result[window_length - 1:] = (
        cumsum[window_length - 1:] - np.concatenate(([0.0], cumsum[:n - window_length]))
    ) / window_length
    return result


def rolling_std(prices, window_length):
    # Var = E[X²] - (E[X])²; NaN for first (window_length-1) entries.
    prices_arr = np.asarray(prices, dtype=float)
    n = len(prices_arr)
    ma = moving_average(prices_arr, window_length)
    cumsum_sq = np.cumsum(prices_arr ** 2)
    mean_sq = (
        cumsum_sq[window_length - 1:] - np.concatenate(([0.0], cumsum_sq[:n - window_length]))
    ) / window_length
    result = np.full(n, np.nan)
    # np.maximum guards against floating-point negatives under the sqrt
    result[window_length - 1:] = np.sqrt(np.maximum(mean_sq - ma[window_length - 1:] ** 2, 0.0))
    return result


def compute_rsi(prices, period=14):
    # Wilder's EMA (alpha = 1/period); loop is irreducible without numba.
    prices_arr = np.asarray(prices, dtype=float)
    n = len(prices_arr)
    deltas = np.diff(prices_arr)

    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    # seed with plain mean over the first window
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs  = avg_gain / np.where(avg_loss == 0, np.finfo(float).eps, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


### PERFORMANCE METRICS

def compute_cagr(portfolio_values, trading_days_per_year=252):
    n_days = len(portfolio_values) - 1
    return (portfolio_values[-1] / portfolio_values[0]) ** (trading_days_per_year / n_days) - 1


def compute_sharpe(daily_returns, risk_free_rate=0.0, trading_days_per_year=252, n_min=30):
    # annualised; NaN if fewer than n_min observations.
    if len(daily_returns) < n_min:
        return np.nan
    excess      = daily_returns - risk_free_rate / trading_days_per_year
    mean_excess = np.sum(excess) / len(excess)
    std_excess  = np.sqrt(np.sum((excess - mean_excess) ** 2) / len(excess))
    if std_excess == 0:
        return np.nan
    return mean_excess / std_excess * np.sqrt(trading_days_per_year)


def compute_max_drawdown(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown    = (portfolio_values - running_max) / running_max
    return np.min(drawdown)


def compute_drawdown_series(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    return (portfolio_values - running_max) / running_max


### SORTINO RATIO

def compute_sortino(daily_returns, target_return=0.0, trading_days_per_year=252, n_min=30):
    # annualised; NaN if fewer than n_min observations.
    if len(daily_returns) < n_min:
        return np.nan
    daily_returns = np.asarray(daily_returns, dtype=float)
    excess        = daily_returns - target_return
    # clamp positive excess to zero — only downside deviation counts
    downside      = np.where(excess < 0, excess, 0.0)
    downside_dev  = np.sqrt(np.sum(downside ** 2) / len(downside))
    if downside_dev == 0:
        return np.nan
    return (np.sum(excess) / len(excess) / downside_dev) * np.sqrt(trading_days_per_year)


### CALMAR RATIO

def compute_calmar(portfolio_values, trading_days_per_year=252):
    portfolio_values = np.asarray(portfolio_values, dtype=float)
    cagr             = compute_cagr(portfolio_values, trading_days_per_year)
    max_dd           = compute_max_drawdown(portfolio_values)
    if max_dd == 0:
        return np.nan
    return cagr / abs(max_dd)


### ANNUAL VOLATILITY

def compute_annual_volatility(daily_returns, trading_days_per_year=252):
    daily_returns = np.asarray(daily_returns, dtype=float)
    n             = len(daily_returns)
    mean_r        = np.sum(daily_returns) / n
    variance      = np.sum((daily_returns - mean_r) ** 2) / n
    return np.sqrt(variance) * np.sqrt(trading_days_per_year)


### PORTFOLIO VALUE TO RETURNS

def pv_to_returns(portfolio_values):
    portfolio_values = np.asarray(portfolio_values, dtype=float)
    return np.concatenate(([0.0], portfolio_values[1:] / portfolio_values[:-1] - 1))


### TRADE EXPECTANCY

def _collect_trade_returns(position_changes_arr, price_returns_arr, trade_cost=0.001):
    # log-returns per completed round-trip; shared by expectancy and win-rate.
    position_changes_arr = np.asarray(position_changes_arr, dtype=float)
    price_returns_arr    = np.asarray(price_returns_arr,    dtype=float)

    trade_returns = []
    in_trade      = False
    log_r         = 0.0

    for i in range(len(position_changes_arr)):
        if position_changes_arr[i] > 0 and not in_trade:
            in_trade = True
            log_r    = 0.0
        elif in_trade:
            r = price_returns_arr[i]
            if not np.isnan(r):
                log_r += np.log1p(r)
            if position_changes_arr[i] < 0:
                net_return = np.expm1(log_r) - 2.0 * trade_cost
                trade_returns.append(net_return)
                in_trade = False
                log_r    = 0.0

    return trade_returns


### MOMENT STATISTICS

def numpy_moments(r):
    r = np.asarray(r, dtype=float)
    n = len(r)
    mu  = np.sum(r) / n
    d   = r - mu
    std = np.sqrt(np.sum(d ** 2) / n)
    if std < 1e-10:
        return 0.0, 3.0
    skew = float(np.sum(d ** 3) / n / std ** 3)
    kurt = float(np.sum(d ** 4) / n / std ** 4)
    return skew, kurt


### DEFLATED SHARPE RATIO

def compute_deflated_sharpe(sharpe, n_trials, n_observations, skewness=0.0, kurtosis=3.0):
    # rational-polynomial normal CDF approximation, no scipy.
    n_obs = max(int(n_observations), 2)

    sr_var = (1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2) / (n_obs - 1)
    sr_var = float(np.maximum(sr_var, 1e-12))
    sr_std = float(np.sqrt(sr_var))

    # inverse normal CDF via rational approximation
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

    n       = max(int(n_trials), 1)
    gamma   = 0.5772156649015329  # Euler-Mascheroni constant
    z1      = _phi_inv(1.0 - 1.0 / n)
    z2      = _phi_inv(1.0 - 1.0 / (n * np.e))
    sr_star = ((1.0 - gamma) * z1 + gamma * z2) * sr_std

    # normal CDF via polynomial approximation
    def _phi(x):
        t = 1.0 / (1.0 + 0.2316419 * float(np.abs(x)))
        d = float(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
        p = 1.0 - d * poly
        return p if x >= 0.0 else 1.0 - p

    z   = (sharpe - sr_star) / sr_std
    dsr = _phi(z)
    return dsr, sr_star


### BASKET SORTINO

def basket_sortino(signal_fn, df_basket, **params):
    # mean Sortino across basket ETFs with 1-day lag.
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
            strat = dr[1:] * arr[:-1]
            s = compute_sortino(strat)
            scores.append(s if s == s else float('nan'))
        except Exception:
            scores.append(float('nan'))
    valid = [s for s in scores if s == s]
    return float(np.mean(valid)) if valid else float('nan')


### IS / OOS SPLIT

def slice_period(df, start, end):
    return df[(df.index >= start) & (df.index <= end)].copy()


### SIGNAL 0 – MOVING AVERAGE CROSSOVER

def ma_signal(series, short_window, long_window):
    # Golden Cross entry, death cross exit.
    prices = np.asarray(series, dtype=float)
    n      = len(prices)

    short_ma = moving_average(prices, short_window)
    long_ma  = moving_average(prices, long_window)

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

def rsi_signal(series, period=14, oversold=30, overbought=70):
    prices = np.asarray(series, dtype=float)

    rsi = compute_rsi(prices, period)

    valid      = ~np.isnan(rsi)
    entry_mask = valid & (rsi < oversold)
    exit_mask  = valid & (rsi > overbought)

    signal     = _vectorised_signal(entry_mask, exit_mask)
    pos_change = np.concatenate(([0.0], signal[1:] - signal[:-1]))

    signals_df = pd.DataFrame(index=series.index)
    signals_df['signal']          = signal
    signals_df['rsi']             = rsi
    signals_df['position_change'] = pos_change
    return signals_df


### SIGNAL 2 – BOLLINGER BANDS MEAN REVERSION

def bollinger_signal(series, window=20, num_std=2):
    # Entry below lower band; exit above middle band.
    prices = np.asarray(series, dtype=float)

    ma         = moving_average(prices, window)
    std        = rolling_std(prices, window)
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std

    valid      = ~np.isnan(lower_band)
    entry_mask = valid & (prices < lower_band)
    exit_mask  = valid & (prices > ma)

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
    # alpha = 2/(span+1), seeded with SMA of the first span observations.
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
    # Buy on MACD line crossing above signal line, sell on the reverse.
    prices = np.asarray(series, dtype=float)

    ema_fast   = exponential_moving_average(prices, fast_span)
    ema_slow   = exponential_moving_average(prices, slow_span)
    macd_line  = ema_fast - ema_slow

    sig_line    = np.full(len(prices), np.nan)
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


### SIGNAL 3 – DONCHIAN CHANNEL BREAKOUT

def donchian_signal(series, entry_window=55, exit_window=20):
    prices = np.asarray(series, dtype=float)
    n      = len(prices)

    from numpy.lib.stride_tricks import sliding_window_view

    entry_high = np.full(n, np.nan)
    if n > entry_window:
        wins = sliding_window_view(prices[:-1], entry_window)
        entry_high[entry_window:] = np.max(wins, axis=1)

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


### BUILD HEATMAP MATRIX

def build_matrix(grid_results, row_vals, col_vals, row_key, col_key):
    mat = np.full((len(row_vals), len(col_vals)), np.nan)
    for params, score in grid_results:
        r = row_vals.index(params[row_key]) if params[row_key] in row_vals else -1
        c = col_vals.index(params[col_key]) if params[col_key] in col_vals else -1
        if r >= 0 and c >= 0 and score == score:
            mat[r, c] = score
    return mat


### VISUALISATION

def draw_heatmap(ax, data, row_labels, col_labels, row_title, col_title,
                 title, star_row, star_col, colorbar_label='Sortino'):
    # ★ marks the chosen params.
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
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if not np.isnan(data[r, c]):
                marker = ' ★' if (r == star_row and c == star_col) else ''
                ax.text(c, r, f'{data[r,c]:.2f}{marker}',
                        ha='center', va='center', fontsize=7, color='black',
                        fontweight='bold' if marker else 'normal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


### PORTFOLIO SIMULATION

def run_portfolio_sim(traded_tickers, df_prices, df_position_changes,
                      df_price_changes, init_cash=1.0):
    # equal-weight, 1-day execution lag; returns per-ticker + cash DataFrame.
    traded_tickers = list(traded_tickers)
    n_etfs = len(traded_tickers)

    raw_pc    = df_position_changes.to_numpy()
    lagged_pc = np.vstack([np.zeros((1, raw_pc.shape[1])), raw_pc[:-1]])
    df_pos_exec = pd.DataFrame(lagged_pc,
                               index=df_position_changes.index,
                               columns=df_position_changes.columns)

    def _open(position, pos_chg):
        total     = float(np.nansum(position))
        target    = total / n_etfs
        vec       = np.maximum(np.array([pos_chg[t] for t in traded_tickers],
                                        dtype=float), 0.0)
        invest    = vec * target
        cash      = position[-1]
        total_buy = float(np.nansum(invest))
        if total_buy > cash and total_buy > 0:
            invest = invest * (cash / total_buy)
        return np.append(invest + position[:-1],
                         cash - float(np.nansum(invest)))

    def _hold(position, price_change):
        return np.concatenate(
            (position[:-1] * price_change[:n_etfs], [position[-1]]))

    def _close(position, pos_chg):
        vec = np.concatenate(
            ([pos_chg[t] < 0.0 for t in traded_tickers], [False]))
        position[-1] += np.sum(position[vec])
        position[vec]  = 0.0
        return position

    rows     = []
    is_first = True
    for idx, pos_chg in df_pos_exec.iterrows():
        if is_first:
            rows.append(_open(
                np.concatenate((np.zeros(n_etfs), [init_cash])), pos_chg))
            is_first = False
        else:
            hp = _hold(rows[-1].copy(),
                       df_price_changes.loc[[idx]].to_numpy()[0])
            hp = _close(hp, pos_chg)
            rows.append(_open(hp, pos_chg))

    return pd.DataFrame(rows,
                        index=df_prices.index,
                        columns=traded_tickers + ['cash']), df_pos_exec


def compute_win_rate(position_changes_arr, price_returns_arr):
    trade_returns = _collect_trade_returns(
        position_changes_arr, price_returns_arr, trade_cost=0.0)
    if len(trade_returns) == 0:
        return np.nan
    wins = int(np.sum(np.asarray(trade_returns, dtype=float) >= 0))
    return wins / len(trade_returns)


### DATA LOADING

def load_etf(tickers_list, csv_name, start, end, data_dir):
    # Load ETF prices from a local CSV cache; downloads via yahooquery if the file is absent.
    # Called by research_notebook to source sector ETF and benchmark price histories.
    import pathlib as _pl
    csv_path = _pl.Path(data_dir) / csv_name
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = df[[t for t in tickers_list if t in df.columns]]
    else:
        df, _ = download_stock_price_data(tickers_list, start, end)
        df.to_csv(csv_path)
        print(f'Downloaded and cached -> {csv_name}.')
    df.index = pd.to_datetime(df.index)
    if df.empty or len(df.columns) == 0:
        raise ValueError(
            f'load_etf: no columns for {tickers_list} in {csv_name}. '
            f'File cols: {list(pd.read_csv(_pl.Path(data_dir) / csv_name, index_col=0, nrows=0).columns)}'
        )
    return df


### PORTFOLIO HELPERS

def basket_portfolio_value(signal_fn, df_basket, params):
    # Equal-weight basket backtest with a 1-day execution lag (signal at t-1 earns return at t).
    # Called by research_notebook to produce cumulative portfolio values for IS/OOS comparison.
    n_stocks = len(df_basket.columns)
    weight   = 1.0 / n_stocks
    returns_matrix = np.zeros((len(df_basket), n_stocks))
    signals_matrix = np.zeros((len(df_basket), n_stocks))
    for j, col in enumerate(df_basket.columns):
        px = df_basket[col].to_numpy(dtype=float)
        dr = np.concatenate(([0.0], px[1:] / px[:-1] - 1))
        returns_matrix[:, j] = dr
        try:
            sig = signal_fn(df_basket[col], **params)
            signals_matrix[:, j] = sig['signal'].to_numpy(dtype=float)
        except Exception:
            pass
    lagged_signals = np.vstack([np.zeros((1, n_stocks)), signals_matrix[:-1]])
    daily_ret = np.sum(lagged_signals * returns_matrix, axis=1) * weight
    return np.cumprod(1.0 + daily_ret)


def spx_normalise(df_basket, df_benchmark):
    # Align the benchmark DataFrame to the basket index (forward-fill) and normalise to 1.0.
    # Used in research_notebook to produce a comparable S&P 500 series for overlay plots.
    aligned = df_benchmark.reindex(df_basket.index, method='ffill')
    col = '^GSPC' if '^GSPC' in aligned.columns else aligned.columns[0]
    v = aligned[col].to_numpy(dtype=float)
    return v / v[0]


def sortino_from_pv(pv):
    # Annualised Sortino ratio directly from a cumulative portfolio-value array.
    # Convenience wrapper used in research_notebook summary tables.
    return compute_sortino(pv_to_returns(pv)[1:])


def compute_dd(pv):
    # Drawdown series as a percentage (e.g. -15.0 means 15 % below the running peak).
    # Used in research_notebook drawdown plots to show strategy vs. S&P 500 underwater curves.
    return compute_drawdown_series(pv) * 100


def run_portfolio(df_p, df_pc, df_pos_changes, init_cash=1.0):
    # Convenience wrapper around run_portfolio_sim; returns the position DataFrame only.
    # Called by assessment_notebook's period_stats to avoid repeating the full keyword-argument list.
    tks = list(df_pos_changes.columns)
    df_pos, _ = run_portfolio_sim(
        traded_tickers=tks, df_prices=df_p[tks],
        df_position_changes=df_pos_changes,
        df_price_changes=df_pc, init_cash=init_cash,
    )
    return df_pos


### SCREENING HELPERS

def screen_backtest(signal_fn, series, params):
    ser = series.dropna()
    px  = ser.to_numpy(dtype=float)
    dr  = np.concatenate(([0.], px[1:] / px[:-1] - 1))
    sig = signal_fn(ser, **params)
    arr = sig['signal'].to_numpy(dtype=float)
    pos = np.concatenate(([0.], arr[:-1]))
    dn  = pos * dr
    return np.cumprod(1. + dn), dn


def screen_optimise(signal_fn, series_is, grid):
    best_s, best_p = -np.inf, None
    for p in grid:
        try:
            _, dn = screen_backtest(signal_fn, series_is, p)
            s = compute_sortino(dn[1:])
            if s == s and s > best_s:
                best_s, best_p = s, dict(p)
        except Exception:
            pass
    return best_p, float(best_s) if best_s > -np.inf else float('nan')


def screen_metrics(pv):
    dr = np.concatenate(([0.], pv[1:] / pv[:-1] - 1))
    return {
        'Sortino': compute_sortino(dr[1:]),
        'Sharpe':  compute_sharpe(dr[1:]),
        'CAGR':    compute_cagr(pv),
        'MaxDD':   compute_max_drawdown(pv),
    }


def screen_eval(df_period, sig_name, tk, screen_opt, signal_grids):
    bp = screen_opt[sig_name][tk]['params']
    if bp is None or tk not in df_period.columns:
        return {k: float('nan') for k in ['Sortino', 'Sharpe', 'CAGR', 'MaxDD']}, None
    ser = df_period[tk].dropna()
    if len(ser) < 60:
        return {k: float('nan') for k in ['Sortino', 'Sharpe', 'CAGR', 'MaxDD']}, None
    try:
        pv, _ = screen_backtest(signal_grids[sig_name][0], ser, bp)
        return screen_metrics(pv), pv
    except Exception:
        return {k: float('nan') for k in ['Sortino', 'Sharpe', 'CAGR', 'MaxDD']}, None
