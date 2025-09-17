import numpy as np
import pandas as pd

def create_alpha_decay_curve(initial_alpha, decay_rate, time_periods, periods_per_year=252):
    """
    Create an exponentially decaying alpha series.

    Parameters
    - initial_alpha: annualized alpha (e.g., 0.05 = 5%/year)
    - decay_rate: annual decay rate lambda (e.g., 0.1)
    - time_periods: number of discrete periods (e.g., trading days)
    - periods_per_year: number of periods per year (default 252 for daily)

    Returns
    - time_index (np.ndarray): integer periods 0..N-1
    - alpha_per_period (np.ndarray): per-period alpha values after scaling
    """
    t_idx = np.arange(time_periods, dtype=float)
    t_years = t_idx / float(periods_per_year)
    # Scale annual alpha to per-period alpha and apply exponential decay in years
    alpha_per_period = (initial_alpha / float(periods_per_year)) * np.exp(-decay_rate * t_years)
    return t_idx, alpha_per_period

def calculate_strategy_returns(benchmark_returns,
                               beta,
                               alpha_values,
                               noise_std_frac: float = 0.25,
                               random_state: int | None = None):
    """
    Strategy returns model:
      r_strategy(t) = alpha(t) + beta * r_benchmark(t) + epsilon(t)

    epsilon ~ Normal(0, noise_std_frac * std(r_benchmark))

    Parameters
    - benchmark_returns: array-like of benchmark percent returns per period
    - beta: market exposure
    - alpha_values: per-period alpha series (same length or longer than returns)
    - noise_std_frac: idiosyncratic volatility as a fraction of benchmark std (default 0.25)
    - random_state: optional RNG seed for reproducibility
    """
    # Coerce inputs to float arrays
    br = np.asarray(benchmark_returns, dtype=float)
    alpha = np.asarray(alpha_values, dtype=float)

    n = min(len(br), len(alpha))
    if n == 0:
        return np.array([], dtype=float)

    br = br[:n]
    alpha = alpha[:n]

    # Idiosyncratic noise tied to benchmark volatility
    rng = np.random.default_rng(random_state)
    bench_std = float(np.nanstd(br)) if np.isfinite(np.nanstd(br)) else 0.0
    eps_std = abs(noise_std_frac) * bench_std
    eps = rng.normal(loc=0.0, scale=eps_std, size=n) if eps_std > 0 else np.zeros(n, dtype=float)

    return alpha + beta * br + eps
    
def calculate_equity_curves(benchmark_returns, strategy_returns, initial_capital):
    # Coerce to float numpy arrays to ensure consistent numeric types
    br = np.asarray(benchmark_returns, dtype=float)
    sr = np.asarray(strategy_returns, dtype=float)

    benchmark_equity = initial_capital * np.cumprod(1.0 + br)
    strategy_equity = initial_capital * np.cumprod(1.0 + sr)
    return benchmark_equity, strategy_equity