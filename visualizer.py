import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

def create_visualization(benchmark_equity,
                         strategy_equity,
                         alpha_time,
                         alpha_values,
                         initial_alpha,
                         decay_rate,
                         dark_mode=False,
                         window=500,
                         max_points=1000):
    # Optional dark mode
    if dark_mode:
        plt.style.use('dark_background')

    # Coerce all inputs to 1D float numpy arrays to avoid dtype issues
    be = np.asarray(benchmark_equity, dtype=float).ravel()
    se = np.asarray(strategy_equity, dtype=float).ravel()
    at = np.asarray(alpha_time, dtype=float).ravel()
    av = np.asarray(alpha_values, dtype=float).ravel()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.set_title('Equity Curves')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3, color='lightgray')
    
    ax2.set_title('Alpha Decay')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Alpha')
    ax2.grid(True, alpha=0.3, color='lightgray')
    
    line1, = ax1.plot([], [], 'b-', label='Benchmark', linewidth=2)
    line2, = ax1.plot([], [], 'r-', label='Strategy', linewidth=2)
    ax1.legend()
    
    decay_line, = ax2.plot(at, av, 'orange', linewidth=2, label='Alpha Decay')
    dot, = ax2.plot([], [], 'ro', markersize=8)
    ax2.legend()
    
    equation_text = f'α(t) = α₀ × e^(-λt)\nα₀ = {initial_alpha:.4f} (Initial Alpha), λ = {decay_rate:.4f} (Decay Rate)'
    fig.text(0.5, 0.02, equation_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    total_len = max(len(be), len(se))
    x_max_left = total_len
    x_max_right = float(np.nanmax(at)) if at.size > 0 and np.isfinite(np.nanmax(at)) else 1.0

    # Currency formatter for equity axis
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
    # Prefer reasonable number of ticks
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    # Percent formatter for alpha axis (per-period values are small)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3%}"))

    def animate(frame):
        current_frame = min(frame, total_len - 1)

        # Determine visible window (pan to the right as we animate)
        if window and isinstance(window, (int, float)) and window > 1:
            start = max(0, current_frame - int(window) + 1)
        else:
            start = 0
        end = current_frame + 1

        be_slice = be[start:end]
        se_slice = se[start:end]
        x_vals = np.arange(len(be_slice))  # start at 0 in the window

        # Optional downsampling for very long windows
        if len(x_vals) > max_points:
            step = max(1, len(x_vals) // max_points)
            x_vals = x_vals[::step]
            be_slice = be_slice[::step]
            se_slice = se_slice[::step]

        line1.set_data(x_vals, be_slice)
        line2.set_data(x_vals, se_slice)

        # Dynamic zoomed-in y-limits based on the currently visible window
        if len(be_slice) > 0 and len(se_slice) > 0:
            cur_min = float(np.nanmin([np.nanmin(be_slice), np.nanmin(se_slice)]))
            cur_max = float(np.nanmax([np.nanmax(be_slice), np.nanmax(se_slice)]))
        elif len(be_slice) > 0:
            cur_min = float(np.nanmin(be_slice)); cur_max = float(np.nanmax(be_slice))
        elif len(se_slice) > 0:
            cur_min = float(np.nanmin(se_slice)); cur_max = float(np.nanmax(se_slice))
        else:
            cur_min, cur_max = 0.0, 1.0

        if not np.isfinite(cur_min) or not np.isfinite(cur_max) or cur_max == cur_min:
            cur_min, cur_max = 0.0, 1.0

        # Ensure some padding and a minimum span so lines are visible
        span = max(cur_max - cur_min, 1e-6)
        pad = max(span * 0.08, max(1.0, cur_max * 0.01))
        y_low = cur_min - pad
        y_high = cur_max + pad

        # X limits reflect the window width for better visibility
        ax1.set_xlim(0, max(10, len(x_vals)))
        ax1.set_ylim(y_low, y_high)

        alpha_frame = min(current_frame, len(av) - 1) if av.size > 0 else 0
        if av.size > 0 and at.size > 0:
            dot.set_data([at[alpha_frame]], [av[alpha_frame]])

        # Dynamic y-limits for alpha using the actual per-period alpha values
        if av.size > 0:
            av_slice = av[:alpha_frame+1] if alpha_frame >= 0 else av[:1]
            av_max = float(np.nanmax(av_slice)) if av_slice.size > 0 else 0.0
            av_min = float(np.nanmin(av_slice)) if av_slice.size > 0 else 0.0
            if not np.isfinite(av_max) or not np.isfinite(av_min) or av_max == av_min:
                av_min, av_max = 0.0, max(1e-6, float(av[0]) if av.size > 0 else 1e-6)
            pad = max((av_max - av_min) * 0.1, av_max * 0.05, 1e-6)
            y0 = min(0.0, av_min - 0.5 * pad)
            y1 = av_max + pad
        else:
            y0, y1 = 0.0, 1.0

        ax2.set_xlim(0, x_max_right)
        ax2.set_ylim(y0, y1)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        return line1, line2, dot
    
    ani = animation.FuncAnimation(fig, animate, frames=total_len, 
                                  interval=50, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    return ani