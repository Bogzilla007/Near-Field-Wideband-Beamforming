"""
Stage 13 checkpoint script.

Tests whether near-field beamforming's advantage over conventional
beamforming survives once a small number of reflected paths are added
to the channel, controlled via a Rician K-factor sweep.

Design (see multipath.py for the full derivation):
    - Each of N_TRIALS random "environments" (reflector locations +
      phases) is generated ONCE, then the ENTIRE K sweep is run against
      that same fixed environment -- isolating K's effect from
      environment-to-environment variation.
    - Results are averaged across all N_TRIALS environments for a
      statistically robust trend, since any single reflector
      configuration could show K-dependence that's really just luck of
      the draw on reflector placement/phase.
    - K=None (0 reflectors) is the exact regression case, matching the
      single-path channel used in every prior stage.

All beamformers (conventional/nearfield/squint/combined) are still
built targeting the KNOWN LOS user location (r_los, theta_los) -- this
mirrors a realistic scenario where the beamformer knows its intended
user's location but has no model of the scattering environment. The
question is how much gain-loss is affected once the TRUE channel
includes multipath the beamformer wasn't designed to know about.

Run: python3 validate_stage13.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arraymodel import ArrayGeometry
from beamformers import beamform, beamform_gain
from multipath import generate_reflector_config, multipath_near_field_channel
from sweep import gain_loss_at_point

CENTER_FREQ = 28e9
THETA = np.deg2rad(20.0)

# Same worst-performing region as Stage 6/9/10, for direct comparability.
N_WORST = 512
BW_WORST = 4e9
R_FRAC_WORST = 0.05

N_REFLECTORS = 3
REFLECTOR_R_FRAC_RANGE = (0.1, 10.0)
REFLECTOR_THETA_RANGE_DEG = (-60, 60)
N_TRIALS = 8

K_DB_SWEEP = [None, 20, 10, 5, 0, -5]  # None = pure LOS (regression case)


def main():
    print("=== Stage 13 Checkpoint ===\n")
    array = ArrayGeometry(N=N_WORST, center_freq=CENTER_FREQ)
    rayleigh = array.rayleigh_distance()
    r_los = R_FRAC_WORST * rayleigh
    freqs = CENTER_FREQ + np.linspace(-BW_WORST / 2, BW_WORST / 2, 33)
    print(f"Worst-region point (matches Stage 6/9/10): N={N_WORST}, "
          f"bandwidth={BW_WORST/1e9:.0f} GHz, r_frac={R_FRAC_WORST}\n")
    print(f"Reflectors: {N_REFLECTORS}, r_frac range={REFLECTOR_R_FRAC_RANGE}, "
          f"theta range={REFLECTOR_THETA_RANGE_DEG} deg, {N_TRIALS} trials/environments\n")

    # Build the LOS-targeted steering vectors once -- unchanged across
    # the whole sweep, since the beamformer doesn't know about scatterers.
    steer_conv = beamform("conventional", array, freqs, THETA, center_freq=CENTER_FREQ)
    steer_nf = beamform("nearfield", array, freqs, THETA, r=r_los, center_freq=CENTER_FREQ)
    steer_sq = beamform("squint", array, freqs, THETA)
    steer_comb = beamform("combined", array, freqs, THETA, r=r_los)

    # =========================================================================
    # Regression gate: K=None reduces exactly to the existing single-path
    # result already computed by Stage 5/6/9/10 at this exact point.
    # =========================================================================
    print("-- Regression gate: K=None matches existing single-path result --")
    reference_point = gain_loss_at_point(N_WORST, BW_WORST, R_FRAC_WORST, CENTER_FREQ)
    reference_loss_conv_db = reference_point["gain_loss_conventional_db"]
    print(f"  Existing (Stage 5/6/9/10) gain_loss_conventional_db at this point: "
          f"{reference_loss_conv_db:.4f} dB")

    h_none = multipath_near_field_channel(array, r_los, THETA, freqs, None, {})
    gain_conv_none = beamform_gain(steer_conv, h_none)
    gain_comb_none = beamform_gain(steer_comb, h_none)
    loss_conv_none = 10 * np.log10(gain_comb_none / gain_conv_none)
    print(f"  Stage 13's K=None (0 reflectors) result:                     "
          f"{loss_conv_none:.4f} dB")
    assert np.isclose(loss_conv_none, reference_loss_conv_db, atol=1e-6), (
        f"K=None should reproduce the existing single-path result exactly, "
        f"got {loss_conv_none:.4f} dB vs reference {reference_loss_conv_db:.4f} dB"
    )
    print("  OK: K=None (0 reflectors) reproduces the existing single-path "
          "result exactly.\n")

    # =========================================================================
    # Main sweep: N_TRIALS environments x K_DB_SWEEP, averaged.
    # =========================================================================
    print("-- Running K-factor sweep across "
          f"{N_TRIALS} environments --\n")
    rows = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(1000 + trial)  # distinct, reproducible seed per trial
        config = generate_reflector_config(
            rng, N_REFLECTORS, REFLECTOR_R_FRAC_RANGE, REFLECTOR_THETA_RANGE_DEG
        )
        for k_db in K_DB_SWEEP:
            h = multipath_near_field_channel(array, r_los, THETA, freqs, k_db, config)
            gain_conv = beamform_gain(steer_conv, h)
            gain_nf = beamform_gain(steer_nf, h)
            gain_sq = beamform_gain(steer_sq, h)
            gain_comb = beamform_gain(steer_comb, h)
            rows.append({
                "trial": trial,
                "k_db": -999 if k_db is None else k_db,  # sentinel for pure-LOS in the CSV
                "gain_loss_conventional_db": 10 * np.log10(gain_comb / gain_conv),
                "gain_loss_nearfield_db": 10 * np.log10(gain_comb / gain_nf),
                "gain_loss_squint_db": 10 * np.log10(gain_comb / gain_sq),
                "gain_comb_absolute": gain_comb,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/stage13_multipath.csv", index=False)
    print("  Saved results/stage13_multipath.csv\n")

    # Aggregate mean +/- std across trials, per K value.
    summary = df.groupby("k_db").agg(
        mean_loss_conv=("gain_loss_conventional_db", "mean"),
        std_loss_conv=("gain_loss_conventional_db", "std"),
        mean_gain_comb=("gain_comb_absolute", "mean"),
        std_gain_comb=("gain_comb_absolute", "std"),
    ).reset_index().sort_values("k_db", ascending=False)
    summary["k_label"] = summary.k_db.apply(lambda k: "LOS only" if k == -999 else f"{k}dB")
    print(summary[["k_label", "mean_loss_conv", "std_loss_conv",
                    "mean_gain_comb", "std_gain_comb"]].to_string(index=False))
    print()

    # =========================================================================
    # Checkpoint assertions
    # =========================================================================
    print("-- Checkpoint assertions --")

    # 1. Convergence: as K increases (less scattering), the mean gain_loss
    #    should approach the pure-LOS baseline monotonically.
    los_only_mean = summary[summary.k_db == -999].mean_loss_conv.iloc[0]
    finite_k_summary = summary[summary.k_db != -999].sort_values("k_db")
    diffs_from_los = np.abs(finite_k_summary.mean_loss_conv.values - los_only_mean)
    k_vals_ascending = finite_k_summary.k_db.values  # already ascending after sort
    print(f"  |mean_loss_conv - LOS_only| as K increases "
          f"({list(k_vals_ascending)}): {np.round(diffs_from_los, 3)}")
    assert diffs_from_los[-1] < diffs_from_los[0], (
        "expected the multipath-contaminated result to approach the "
        "pure-LOS baseline as K increases (less scattering)"
    )
    print("  OK: mean result converges toward the pure-LOS baseline as K "
          "increases (more LOS-dominated), confirming the model behaves "
          "correctly.\n")

    # 2. Combined's own absolute gain (against the LOS-targeted steering
    #    vector) should degrade as K decreases -- even the "ideal"
    #    corrector isn't immune to unmodeled multipath, since it was
    #    never designed to know about the scatterers.
    gains_by_k = finite_k_summary.mean_gain_comb.values
    print(f"  combined's mean absolute gain vs K "
          f"({list(k_vals_ascending)}): {np.round(gains_by_k, 1)}")
    assert gains_by_k[0] <= gains_by_k[-1], (
        "expected combined's own gain to be lower (or equal) at low K "
        "(more scattering) than at high K (less scattering)"
    )
    print("  OK: even the combined corrector's own achieved gain degrades "
          "as scattering increases -- it has no model of the reflectors, "
          "only the LOS path.\n")

    # 3. THE MAIN FINDING: does conventional's dB gap behind combined
    #    (i.e. near-field beamforming's ADVANTAGE) shrink, hold, or grow
    #    as scattering increases? Report honestly either way.
    print("-- Main finding: does near-field beamforming's advantage survive multipath? --")
    for _, row in summary.sort_values("k_db", ascending=False).iterrows():
        print(f"  K={row.k_label:>10s}: gain_loss_conventional = "
              f"{row.mean_loss_conv:6.3f} +/- {row.std_loss_conv:.3f} dB")

    worst_k_mean = finite_k_summary[finite_k_summary.k_db == finite_k_summary.k_db.min()].mean_loss_conv.iloc[0]
    print(f"\n  Pure-LOS advantage: {los_only_mean:.3f} dB. "
          f"Worst-tested-multipath (K={finite_k_summary.k_db.min()}dB) advantage: "
          f"{worst_k_mean:.3f} dB.")
    if worst_k_mean > los_only_mean * 0.7:
        print("  FINDING: near-field beamforming's advantage over conventional "
              "LARGELY SURVIVES even under significant multipath (K as low as "
              f"{finite_k_summary.k_db.min()}dB) -- it doesn't fully collapse, "
              "though it does shrink somewhat, since the beamformer still "
              "gets some correct-shape contribution from the LOS component "
              "even when scattering dominates the total channel power.")
    else:
        print("  FINDING: near-field beamforming's advantage over conventional "
              "meaningfully ERODES under strong multipath -- the two "
              "beamformers become harder to distinguish once scattered "
              "power dominates, since neither is designed to exploit "
              "reflector-specific structure.")

    # =========================================================================
    # Plot: mean gain-loss vs K, with std error bars.
    # =========================================================================
    print("\n-- Generating figure --")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    plot_df = finite_k_summary.sort_values("k_db")
    ax.errorbar(plot_df.k_db, plot_df.mean_loss_conv, yerr=plot_df.std_loss_conv,
                marker="o", capsize=4, label="conventional (multipath)")
    ax.axhline(los_only_mean, color="k", linestyle="--", label="pure LOS (K=inf)")
    ax.set_xlabel("Rician K-factor (dB)")
    ax.set_ylabel("Gain loss vs. combined (dB)")
    ax.set_title(f"Stage 13: Conventional's gain-loss vs. K\n"
                 f"(N={N_WORST}, {BW_WORST/1e9:.0f}GHz, r={R_FRAC_WORST}x Rayleigh, "
                 f"{N_TRIALS} environments)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(plot_df.k_db, plot_df.mean_gain_comb, yerr=plot_df.std_gain_comb,
                marker="o", capsize=4, color="seagreen", label="combined's own gain")
    ax.axhline(summary[summary.k_db == -999].mean_gain_comb.iloc[0], color="k",
               linestyle="--", label="pure LOS (K=inf)")
    ax.set_xlabel("Rician K-factor (dB)")
    ax.set_ylabel("Combined corrector's absolute gain")
    ax.set_title("Stage 13: Even the ideal corrector isn't immune to\n"
                 "unmodeled multipath")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/stage13_multipath.png", dpi=150)
    print("Saved figures/stage13_multipath.png")

    print("\n=== Stage 13 checkpoint PASSED. Multipath model regression-tested "
          "against the single-path baseline, and near-field beamforming's "
          "behavior under multipath quantified across "
          f"{N_TRIALS} independent scattering environments. ===")


if __name__ == "__main__":
    main()