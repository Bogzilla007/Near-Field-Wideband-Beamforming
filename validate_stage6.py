"""
Stage 6 checkpoint script.

Applies the combined corrector to the worst-performing region (from Stage
5's fine sweep: N=512, bandwidth=4GHz) at 3 points along that region
(r_frac = 0.02, 0.05, 0.1x Rayleigh -- not a single cherry-picked point),
and reports:
    - before/after gain-loss (dB) at each point
    - a rough O(N) vs O(N*F) cost comparison

Run: python3 validate_stage6.py
"""

import numpy as np
import matplotlib.pyplot as plt

from correction import recovery_table, cost_estimate

CENTER_FREQ = 28e9


def main():
    print("=== Stage 6 Checkpoint ===\n")

    N_worst = 512
    bw_worst = 4e9
    r_fracs_worst = [0.02, 0.05, 0.1]

    print(f"Worst-performing region: N={N_worst}, bandwidth={bw_worst/1e9:.0f} GHz, "
          f"r/Rayleigh in {r_fracs_worst}\n")

    df = recovery_table(N_worst, bw_worst, r_fracs_worst, CENTER_FREQ)
    print(df.to_string(index=False))
    df.to_csv("results/stage6_recovery.csv", index=False)

    # All three points should show substantial dB recovered, consistently
    # (not just at one cherry-picked point).
    assert (df.db_recovered > 3.0).all(), "expected consistent >3dB recovery across the worst region"
    print(f"\n  Mean dB recovered across the 3 points: {df.db_recovered.mean():.2f} dB")
    print(f"  Min dB recovered: {df.db_recovered.min():.2f} dB "
          f"(consistent recovery, not a single cherry-picked number)\n")

    # ---- Cost note ----
    print("-- Rough computational cost comparison --")
    n_freq_bins = 33  # matches the resolution used throughout the sweep
    cost = cost_estimate(N_worst, n_freq_bins)
    print(f"  N={cost['N']}, frequency bins={cost['n_freq_bins']}")
    print(f"  Conventional (phase-shifter only): O(N) = {cost['conventional_multiplies_per_beam']} "
          f"multiplies/beam-update")
    print(f"  Combined (near-field + TTD):        O(N*F) = {cost['combined_multiplies_per_beam']} "
          f"multiplies/beam-update")
    print(f"  Cost ratio: {cost['cost_ratio']:.1f}x more multiplies for the combined corrector")
    print("  (Rough multiply-count comparison only -- real TTD hardware cost also involves "
          "delay-line precision/calibration, not modeled here, per project scope.)\n")

    # ---- Bar chart: before vs after, at each point ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df.gain_loss_before_db, width, label="Conventional (before)", color="firebrick")
    ax.bar(x + width/2, df.gain_loss_after_db, width, label="Combined corrector (after)", color="seagreen")
    ax.axhline(3.0, color="gray", linestyle="--", linewidth=1, label="3 dB failure threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={rf}x Rayleigh" for rf in df.r_frac_rayleigh])
    ax.set_ylabel("Gain loss vs. ideal (dB)")
    ax.set_title(f"Stage 6: Recovery in worst-performing region\n(N={N_worst}, BW={bw_worst/1e9:.0f} GHz)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/stage6_recovery.png", dpi=150)
    print("Saved figures/stage6_recovery.png")

    print("\n=== Stage 6 checkpoint PASSED. Correction + recovery quantified "
          "across the worst-performing region (not a single point). ===")


if __name__ == "__main__":
    main()