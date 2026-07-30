"""Analyze the gating-analysis log produced by main_threaded.py.

Produces 8 subplots + console summary:
  1. Position uncertainty evolution (3σ per axis)
  2. Condition number (covariance anisotropy)
  3. Optimization health (avg error/factor) with alternating pattern detection
  4. Motion profile (median pixel displacement)
  5. Observations per frame
  6. Factor graph growth + rate
  7. Scatter: displacement vs uncertainty
  8. Scatter: observations vs error

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

    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]

    # Detect whether new columns are present
    has_ate = "ate_rmse" in df.columns
    has_landmarks = "n_landmarks" in df.columns
    has_depth = "df_attempted" in df.columns
    has_worst_dir = "worst_dir_x" in df.columns

    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex="col")
    fig.suptitle("VIO Gating Analysis", fontsize=14, fontweight="bold")

    # Shared vertical cursor across all subplots on hover
    vlines = []
    for ax_row in axes:
        for ax in ax_row:
            vl = ax.axvline(x=0, color='gray', ls='--', lw=0.7, alpha=0, visible=False)
            vlines.append(vl)

    def on_mouse_move(event):
        if event.inaxes is None:
            for vl in vlines:
                vl.set_alpha(0)
            fig.canvas.draw_idle()
            return
        x = event.xdata
        for vl in vlines:
            vl.set_xdata([x, x])
            vl.set_alpha(0.6)
            vl.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # --- 1. Position covariance eigenvalues (3σ) ---
    ax = axes[0, 0]
    for col, label, color in [("cov_eig0", "λ₀ (best)", "tab:green"),
                               ("cov_eig1", "λ₁ (mid)", "tab:blue"),
                               ("cov_eig2", "λ₂ (worst)", "tab:red")]:
        ax.plot(df["t"], np.sqrt(df[col]) * 3, label=f"3σ {label}", color=color, lw=0.8)
    ax.set_ylabel("Position 3σ (m)")
    ax.set_title("Position Uncertainty Eigenvalues")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 2. Condition number ---
    ax = axes[0, 1]
    ax.plot(df["t"], df["condition_number"], color="tab:purple", lw=0.8)
    ax.set_ylabel("Condition Number")
    ax.set_title("Covariance Anisotropy (max/min eigenvalue)")
    ax.grid(True, alpha=0.3)

    # --- 3. Avg error per factor ---
    ax = axes[1, 0]
    ax.plot(df["t"], df["avg_error_per_factor"], color="tab:orange", lw=0.8)
    ax.set_ylabel("Avg Error / Factor")
    ax.set_title("Optimization Health")
    ax.grid(True, alpha=0.3)
    if len(df) > 40:
        errors = df["avg_error_per_factor"].values[30:]
        even_avg = errors[::2].mean()
        odd_avg = errors[1::2].mean()
        ax.axhline(y=even_avg, color="blue", ls=":", alpha=0.5, label=f"Even avg: {even_avg:.2f}")
        ax.axhline(y=odd_avg, color="red", ls=":", alpha=0.5, label=f"Odd avg: {odd_avg:.2f}")
        ax.legend(fontsize=8)

    # --- 4. Median pixel displacement ---
    ax = axes[1, 1]
    ax.plot(df["t"], df["median_pixel_disp"], color="tab:cyan", lw=0.8)
    ax.axhline(y=1.0, color="red", ls="--", alpha=0.5, label="Static threshold (1px)")
    ax.set_ylabel("Median Disp (px)")
    ax.set_title("Motion Profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 5. Observations per frame ---
    ax = axes[2, 0]
    ax.plot(df["t"], df["n_observations"], color="tab:brown", lw=0.8)
    ax.set_ylabel("# Observations")
    ax.set_title("Frontend Observations per Frame")
    ax.grid(True, alpha=0.3)

    # --- 6. Factor graph growth + rate ---
    ax = axes[2, 1]
    ax.plot(df["t"], df["n_factors"], color="tab:olive", lw=0.8)
    ax.set_ylabel("Total Factors")
    ax.set_title("Factor Graph Size")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    factor_rate = df["n_factors"].diff() / df["t"].diff().replace(0, np.nan)
    ax2.plot(df["t"], factor_rate, color="tab:red", lw=0.4, alpha=0.4)
    ax2.set_ylabel("Factors/sec", color="tab:red")

    # --- 7. ATE + Landmarks (if available) ---
    ax = axes[3, 0]
    if has_ate:
        ate_valid = df[df["ate_rmse"] != ""].copy()
        if len(ate_valid) > 0:
            ate_valid["ate_rmse"] = pd.to_numeric(ate_valid["ate_rmse"], errors="coerce")
            ate_valid["frame_error"] = pd.to_numeric(ate_valid["frame_error"], errors="coerce")
            ax.plot(ate_valid["t"], ate_valid["ate_rmse"], color="tab:red", lw=1.0, label="ATE RMSE")
            ax.plot(ate_valid["t"], ate_valid["frame_error"], color="tab:orange", lw=0.5, alpha=0.6, label="Frame error")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "ATE not logged\n(re-run pipeline)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_ylabel("Error (m)")
    ax.set_title("Trajectory Error (ATE)")
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    if has_landmarks:
        ax.plot(df["t"], df["n_landmarks"], color="tab:green", lw=0.8, label="Initialized")
        ax.plot(df["t"], df["n_buffered"], color="tab:gray", lw=0.8, alpha=0.6, label="Buffered")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Landmarks not logged\n(re-run pipeline)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_ylabel("# Landmarks")
    ax.set_title("Landmark Count (initialized vs buffered)")
    ax.grid(True, alpha=0.3)

    # --- 9. Depth filter + worst direction ---
    ax = axes[4, 0]
    if has_depth:
        ax.bar(df["t"], df["df_promoted"], width=0.4, color="tab:green", alpha=0.7, label="Promoted")
        ax.bar(df["t"], df["df_outlier"], width=0.4, bottom=df["df_promoted"],
               color="tab:red", alpha=0.7, label="Outlier")
        ax.bar(df["t"], df["df_pending"], width=0.4,
               bottom=df["df_promoted"] + df["df_outlier"],
               color="tab:gray", alpha=0.5, label="Pending")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Depth filter not logged\n(re-run pipeline)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Depth Filter: Promotion / Outlier / Pending")
    ax.grid(True, alpha=0.3)

    ax = axes[4, 1]
    if has_worst_dir:
        ax.plot(df["t"], df["worst_dir_x"].abs(), color="tab:red", lw=0.8, label="|X|")
        ax.plot(df["t"], df["worst_dir_y"].abs(), color="tab:green", lw=0.8, label="|Y|")
        ax.plot(df["t"], df["worst_dir_z"].abs(), color="tab:blue", lw=0.8, label="|Z|")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Eigenvectors not logged\n(re-run pipeline)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Component|")
    ax.set_title("Worst-Constrained Direction (world frame)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "gating_analysis_plots.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    # ===== Console summary =====
    print("\n" + "=" * 60)
    print("GATING ANALYSIS SUMMARY")
    print("=" * 60)

    static_frames = df[df["median_pixel_disp"] < 1.0]
    moving_frames = df[df["median_pixel_disp"] >= 1.0]
    print(f"\nFrames: {len(df)} total, {len(static_frames)} static, {len(moving_frames)} moving")
    print(f"Duration: {df['t'].iloc[-1]:.1f}s")

    print(f"\n--- Uncertainty ---")
    print(f"Final 3σ worst axis:  {np.sqrt(df['cov_eig2'].iloc[-1])*3:.4f} m")
    print(f"Final 3σ best axis:   {np.sqrt(df['cov_eig0'].iloc[-1])*3:.4f} m")
    print(f"Final condition #:    {df['condition_number'].iloc[-1]:.2f}")
    print(f"Max condition #:      {df['condition_number'].max():.2f} (frame {df.loc[df['condition_number'].idxmax(), 'frame_idx']})")

    print(f"\n--- Optimization ---")
    print(f"Avg error/factor:     {df['avg_error_per_factor'].mean():.3f}")
    print(f"Median error/factor:  {df['avg_error_per_factor'].median():.3f}")
    worst_idx = df["avg_error_per_factor"].idxmax()
    print(f"Max error/factor:     {df['avg_error_per_factor'].max():.3f} (frame {df.loc[worst_idx, 'frame_idx']})")

    if len(df) > 40:
        errors = df["avg_error_per_factor"].values[30:]
        even_avg = errors[::2].mean()
        odd_avg = errors[1::2].mean()
        ratio = max(even_avg, odd_avg) / max(min(even_avg, odd_avg), 1e-10)
        print(f"\n--- Alternating Error Pattern ---")
        print(f"Even-indexed avg:  {even_avg:.3f}")
        print(f"Odd-indexed avg:   {odd_avg:.3f}")
        print(f"Ratio:             {ratio:.2f}x")
        if ratio > 1.3:
            print("  ⚠ Significant alternating pattern — likely relinearization on every-other update")

    print(f"\n--- Motion ---")
    if len(moving_frames) > 0:
        print(f"Median displacement:  {moving_frames['median_pixel_disp'].median():.1f} px")
    print(f"Max displacement:     {df['median_pixel_disp'].max():.1f} px (frame {df.loc[df['median_pixel_disp'].idxmax(), 'frame_idx']})")

    print(f"\n--- Factor Graph ---")
    print(f"Final size:           {df['n_factors'].iloc[-1]} factors")
    print(f"Avg observations/frame: {df['n_observations'].mean():.0f}")

    print(f"\n--- Worst-Constrained Direction ---")
    eig_ratio = df["cov_eig2"].iloc[-1] / df["cov_eig0"].iloc[-1]
    print(f"Final worst/best eigenvalue ratio: {eig_ratio:.2f}")
    if has_worst_dir:
        wd = df[["worst_dir_x", "worst_dir_y", "worst_dir_z"]].iloc[-1]
        abs_wd = wd.abs()
        dominant = abs_wd.idxmax().replace("worst_dir_", "").upper()
        print(f"Final worst direction: [{wd['worst_dir_x']:.3f}, {wd['worst_dir_y']:.3f}, {wd['worst_dir_z']:.3f}]")
        print(f"Dominant axis: {dominant} (|component| = {abs_wd.max():.3f})")
    if len(df) >= 20:
        late = df.iloc[-20:]
        slope0 = np.polyfit(late["t"], late["cov_eig0"], 1)[0]
        slope2 = np.polyfit(late["t"], late["cov_eig2"], 1)[0]
        print(f"Late-stage trends: best axis {slope0:+.2e}/s, worst axis {slope2:+.2e}/s")
        if slope2 > 3 * abs(slope0) and slope2 > 0:
            print("  ⚠ Worst axis growing much faster — possible unobservable direction!")

    if has_landmarks:
        print(f"\n--- Landmarks ---")
        print(f"Final initialized:    {df['n_landmarks'].iloc[-1]}")
        print(f"Final buffered:       {df['n_buffered'].iloc[-1]}")
        print(f"Max initialized:      {df['n_landmarks'].max()}")

    if has_depth:
        print(f"\n--- Depth Filter (totals) ---")
        print(f"Total attempted:      {df['df_attempted'].sum()}")
        print(f"Total promoted:       {df['df_promoted'].sum()}")
        print(f"Total outlier:        {df['df_outlier'].sum()}")
        promo_rate = df['df_promoted'].sum() / max(df['df_attempted'].sum(), 1) * 100
        print(f"Promotion rate:       {promo_rate:.1f}%")

    if has_ate:
        ate_valid = df[df["ate_rmse"] != ""]
        if len(ate_valid) > 0:
            ate_vals = pd.to_numeric(ate_valid["ate_rmse"], errors="coerce").dropna()
            if len(ate_vals) > 0:
                print(f"\n--- ATE ---")
                print(f"Final ATE RMSE:       {ate_vals.iloc[-1]:.4f} m")
                print(f"Min ATE RMSE:         {ate_vals.min():.4f} m")
                print(f"Max ATE RMSE:         {ate_vals.max():.4f} m")

    plt.show()


if __name__ == "__main__":
    main()
