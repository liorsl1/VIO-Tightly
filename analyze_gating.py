"""Analyze the gating-analysis log produced by main_threaded.py.

Loads gating_analysis_log.csv and inspects the relationship between the
geometric triangulation baseline (pixel displacement / parallax angle) and the
factor-graph conditioning (position covariance trace, condition number).

Usage:
    python analyze_gating.py [path/to/gating_analysis_log.csv]
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gating_analysis_log.csv"
    )
    if not os.path.exists(csv_path):
        print(f"Log file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(df.describe(include="all").transpose())

    # Drivers (geometry) vs. responses (conditioning)
    drivers = ["median_pixel_disp", "parallax_max_deg", "parallax_median_deg"]
    responses = ["cov_trace", "condition_number", "avg_error_per_factor"]

    # --- Correlation table (Pearson + Spearman) ---
    print("\n=== Correlations (driver -> response) ===")
    print(f"{'driver':>20} {'response':>22} {'pearson':>9} {'spearman':>9}")
    for d in drivers:
        for r in responses:
            sub = df[[d, r]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 3 or sub[d].std() == 0 or sub[r].std() == 0:
                pear = spear = float("nan")
            else:
                pear = sub[d].corr(sub[r], method="pearson")
                spear = sub[d].corr(sub[r], method="spearman")
            print(f"{d:>20} {r:>22} {pear:>9.3f} {spear:>9.3f}")

    # --- Scatter matrix: each driver vs each response ---
    fig, axes = plt.subplots(
        len(drivers), len(responses),
        figsize=(4 * len(responses), 3.2 * len(drivers)),
        squeeze=False,
    )
    for di, d in enumerate(drivers):
        for ri, r in enumerate(responses):
            ax = axes[di][ri]
            sub = df[[d, r]].replace([np.inf, -np.inf], np.nan).dropna()
            ax.scatter(sub[d], sub[r], s=12, alpha=0.6)
            ax.set_xlabel(d)
            ax.set_ylabel(r)
            if r in ("condition_number", "cov_trace") and (sub[r] > 0).all() and len(sub) > 0:
                ax.set_yscale("log")
    fig.suptitle("Geometry (parallax) vs. graph conditioning")
    fig.tight_layout()

    # --- Time series: parallax and conditioning over frames ---
    fig2, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(df["frame_idx"], df["median_pixel_disp"], color="tab:blue", label="median_pixel_disp")
    ax1.plot(df["frame_idx"], df["parallax_max_deg"], color="tab:green", label="parallax_max_deg")
    ax1.set_xlabel("frame_idx")
    ax1.set_ylabel("pixels / degrees")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(df["frame_idx"], df["condition_number"], color="tab:red", alpha=0.6, label="condition_number")
    ax2.set_yscale("log")
    ax2.set_ylabel("condition number (log)")
    ax2.legend(loc="upper right")
    fig2.suptitle("Per-frame parallax vs. condition number")
    fig2.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    p1 = os.path.join(out_dir, "gating_scatter.png")
    p2 = os.path.join(out_dir, "gating_timeseries.png")
    fig.savefig(p1, dpi=120)
    fig2.savefig(p2, dpi=120)
    print(f"\nSaved plots:\n  {p1}\n  {p2}")
    plt.show()


if __name__ == "__main__":
    main()
