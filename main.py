from data import get_data
from math_models import create_alpha_decay_curve, calculate_strategy_returns, calculate_equity_curves
from visualizer import create_visualization

def get_user_inputs():
    initial_capital = float(input("Enter initial capital: $"))
    ticker = input("Enter ticker symbol for benchmark (e.g., SPY): ").upper()
    years = int(input("Enter number of years of data to pull: "))
    initial_alpha = float(input("Enter initial alpha (e.g., 0.05): "))
    decay_rate = float(input("Enter alpha decay rate (e.g., 0.1): "))
    beta = float(input("Enter strategy beta (market exposure, e.g., 1.0): "))
    noise_std_frac = float(input("Idiosyncratic noise (as fraction of benchmark vol, e.g., 0.5): "))
    seed_inp = input("Random seed (blank for random): ").strip()
    rng_seed = int(seed_inp) if seed_inp else None
    dark_mode_input = input("Enable dark mode? (y/n): ").strip().lower()
    dark_mode = dark_mode_input.startswith('y')
    
    return initial_capital, ticker, years, initial_alpha, decay_rate, beta, noise_std_frac, rng_seed, dark_mode

def main():
    initial_capital, ticker, years, initial_alpha, decay_rate, beta, noise_std_frac, rng_seed, dark_mode = get_user_inputs()
    
    prices, benchmark_returns = get_data(ticker, years)
    
    alpha_time, alpha_values = create_alpha_decay_curve(initial_alpha, decay_rate, len(benchmark_returns))
    
    strategy_returns = calculate_strategy_returns(benchmark_returns, beta, alpha_values, noise_std_frac=noise_std_frac, random_state=rng_seed)
    
    benchmark_equity, strategy_equity = calculate_equity_curves(benchmark_returns, strategy_returns, initial_capital)
    
    animation = create_visualization(benchmark_equity, strategy_equity, alpha_time, alpha_values, initial_alpha, decay_rate, dark_mode=dark_mode)

if __name__ == "__main__":
    main()