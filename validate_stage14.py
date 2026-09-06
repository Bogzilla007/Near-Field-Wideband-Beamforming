"""
Stage 14 checkpoint script.

Tests the fully-connected hybrid beamforming construction (hybrid.py)
at Stage 6's worst-performing region (N=512, 4GHz, r_frac=0.05), across
a power-of-2 sweep of N_RF (RF chain count) from 1 to N=512.

Hard gates (same discipline as every prior stage):
    1. Regression: N_RF=1 must exactly match the existing 'nearfield'
       mode's gain-loss (both are curvature-corrected, frequency-flat,
       single-beam constructions -- see hybrid.py's anchor-selection fix
       for why this wasn't true before a small indexing bug was caught).
    2. Convergence: N_RF=N must recover the fully-digital 'combined'
       corrector's gain exactly (0 dB loss) -- solving an invertible
       N x N linear system exactly, not approximately.
    3. Monotonicity: more RF chains (more analog+digital design freedom)
       should never make gain-loss WORSE.

Main finding tested (not assumed): analog phase shifters can only ever
contribute PHASE, never amplitude -- does the near-field channel's
amplitude taper (1/r_n) require many more RF chains to reconstruct than
the frequency/squint-correction part of the problem does? Found while
building this stage: NO -- convergence to ~0 dB happens by N_RF~16-32,
far short of N=512, because (consistent with Stage 11's finding) the
amplitude taper is a SMOOTH, slowly-varying function across elements at
this array/distance regime, and a modest number of phase-only basis
beams can reconstruct a smooth function via interference, without
needing anywhere near N independent degrees of freedom.

Run: python3 validate_stage14.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arraymodel import ArrayGeometry
from channel import near_field_channel
from beamformers import beamform, beamform_gain
from hybrid import hybrid_effective_steering, cost_estimate_hybrid

CENTER_FREQ = 28e9
THETA = np.deg2rad(20.0)

N_WORST = 512
BW_WORST = 4e9
R_FRAC_WORST = 0.05
N_FREQ_BINS = 33

N_RF_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    N, F = channel.shape
    power = np.sum(np.abs(channel) ** 2)
    return channel * np.sqrt(N * F / power)


def main():
    print("=== Stage 14 Checkpoint ===\n")
    array = ArrayGeometry(N=N_WORST, center_freq=CENTER_FREQ)
    rayleigh = array.rayleigh_distance()
    r = R_FRAC_WORST * rayleigh
    freqs = CENTER_FREQ + np.linspace(-BW_WORST / 2, BW_WORST / 2, N_FREQ_BINS)
    true_channel = _normalize_channel(near_field_channel(array, r, THETA, freqs))

    print(f"Worst-region point (matches Stage 6/9/10/13): N={N_WORST}, "
          f"bandwidth={BW_WORST/1e9:.0f} GHz, r_frac={R_FRAC_WORST}\n")

    gain_combined = beamform_gain(
        beamform("combined", array, freqs, THETA, r=r), true_channel
    )
    gain_conventional = beamform_gain(
        beamform("conventional", array, freqs, THETA, center_freq=CENTER_FREQ), true_channel
    )
    gain_nearfield = beamform_gain(
        beamform("nearfield", array, freqs, THETA, r=r, center_freq=CENTER_FREQ), true_channel
    )
    loss_conventional_db = 10 * np.log10(gain_combined / gain_conventional)
    loss_nearfield_db = 10 * np.log10(gain_combined / gain_nearfield)
    print(f"  Reference: conventional loss vs combined = {loss_conventional_db:.4f} dB")
    print(f"  Reference: nearfield (N_RF=1 target) loss vs combined = {loss_nearfield_db:.4f} dB\n")

    # =========================================================================
    # Sweep N_RF, compute gain-loss and cost at each point.
    # =========================================================================
    print("-- N_RF sweep --")
    rows = []
    for n_rf in N_RF_SWEEP:
        a_eff = hybrid_effective_steering(array, r, THETA, freqs, n_rf)
        gain_hybrid = beamform_gain(a_eff, true_channel)
        loss_db = 10 * np.log10(gain_combined / gain_hybrid)
        cost = cost_estimate_hybrid(N_WORST, n_rf, N_FREQ_BINS)
        rows.append({
            "n_rf": n_rf,
            "gain_loss_db": loss_db,
            "analog_phase_shifters": cost["analog_phase_shifters"],
            "digital_multiplies_per_update": cost["digital_multiplies_per_update"],
        })
        print(f"  N_RF={n_rf:4d}: gain_loss={loss_db:9.4f} dB   "
              f"analog_shifters={cost['analog_phase_shifters']:7d}   "
              f"digital_multiplies/update={cost['digital_multiplies_per_update']:6d}")

    df = pd.DataFrame(rows)
    df.to_csv("results/stage14_hybrid.csv", index=False)
    print("\n  Saved results/stage14_hybrid.csv\n")

    # =========================================================================
    # Checkpoint assertions
    # =========================================================================
    print("-- Checkpoint assertions --")

    # 1. Regression: N_RF=1 matches 'nearfield' mode exactly.
    loss_n_rf_1 = df[df.n_rf == 1].iloc[0].gain_loss_db
    print(f"  N_RF=1 gain_loss = {loss_n_rf_1:.4f} dB, "
          f"'nearfield' mode reference = {loss_nearfield_db:.4f} dB")
    assert np.isclose(loss_n_rf_1, loss_nearfield_db, atol=1e-2), (
        f"N_RF=1 should exactly match 'nearfield' mode's gain-loss "
        f"(both are curvature-corrected, frequency-flat), got "
        f"{loss_n_rf_1:.4f} dB vs {loss_nearfield_db:.4f} dB"
    )
    print("  OK: N_RF=1 exactly reproduces 'nearfield' mode's gain-loss.\n")

    # 2. Convergence: N_RF=N recovers 'combined' mode's gain exactly.
    loss_n_rf_max = df[df.n_rf == N_WORST].iloc[0].gain_loss_db
    print(f"  N_RF={N_WORST} (=N) gain_loss = {loss_n_rf_max:.6f} dB (expect ~0)")
    assert loss_n_rf_max < 0.01, (
        f"N_RF=N should recover the fully-digital combined corrector's "
        f"gain exactly, got {loss_n_rf_max:.6f} dB loss"
    )
    print("  OK: N_RF=N exactly recovers the fully-digital 'combined' "
          "corrector's gain (solves an invertible N x N system exactly).\n")

    # 3. Monotonicity: more RF chains should never make things worse.
    sorted_df = df.sort_values("n_rf")
    losses = sorted_df.gain_loss_db.values
    assert np.all(np.diff(losses) <= 1e-6), (
        f"gain-loss should never increase as N_RF grows, got {losses}"
    )
    print("  OK: gain-loss decreases (or holds) monotonically as N_RF "
          "increases -- more RF chains never hurts.\n")

    # 4. THE MAIN FINDING: where does convergence actually saturate?
    print("-- Main finding: how many RF chains does 'nearly ideal' actually need? --")
    near_ideal = df[df.gain_loss_db < 0.1]
    min_n_rf_near_ideal = int(near_ideal.n_rf.min()) if len(near_ideal) > 0 else None
    print(f"  Minimum N_RF for gain_loss < 0.1 dB (essentially fully-digital "
          f"performance): {min_n_rf_near_ideal} (out of N={N_WORST})")
    assert min_n_rf_near_ideal is not None and min_n_rf_near_ideal <= N_FREQ_BINS, (
        f"expected convergence to essentially-ideal performance well before "
        f"N_RF reaches N (specifically, at or below n_freq_bins={N_FREQ_BINS}), "
        f"got {min_n_rf_near_ideal}"
    )
    print(f"\n  FINDING: essentially-ideal (< 0.1 dB loss) performance is reached "
          f"at N_RF={min_n_rf_near_ideal}, roughly matching n_freq_bins="
          f"{N_FREQ_BINS} -- NOT anywhere near N={N_WORST}. This contradicts "
          f"the initial hypothesis (made while planning this stage) that "
          f"amplitude-taper correction -- which analog phase-only shifters "
          f"cannot directly provide -- would need far more RF chains than "
          f"frequency/squint correction alone. Instead, a modest number of "
          f"phase-only analog beams can jointly reconstruct the near-field "
          f"amplitude taper too, via constructive/destructive interference, "
          f"BECAUSE (consistent with Stage 11's finding) the amplitude taper "
          f"is a smooth, slowly-varying function across elements at this "
          f"array/distance regime (aperture << r) -- it doesn't take many "
          f"basis functions to approximate a smooth function well. "
          f"Practical takeaway: fully-connected hybrid beamforming with as "
          f"few as {min_n_rf_near_ideal} RF chains (out of {N_WORST} "
          f"elements) can recover essentially all of the fully-digital "
          f"combined corrector's benefit at this operating point.\n")

    # =========================================================================
    # Plot: gain-loss vs N_RF (log-x), and cost tradeoff.
    # =========================================================================
    print("-- Generating figure --")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.semilogx(sorted_df.n_rf, sorted_df.gain_loss_db, marker="o", base=2)
    ax.axhline(loss_conventional_db, color="firebrick", linestyle="--",
               label=f"conventional ({loss_conventional_db:.1f} dB)")
    ax.axhline(0.1, color="gray", linestyle=":", label="0.1 dB threshold")
    ax.set_xlabel("Number of RF chains (N_RF)")
    ax.set_ylabel("Gain loss vs. fully-digital combined (dB)")
    ax.set_title("Stage 14: Hybrid beamforming gain-loss vs. N_RF")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.loglog(sorted_df.n_rf, sorted_df.digital_multiplies_per_update, marker="o",
              label="hybrid digital multiplies/update", base=2)
    ax.axhline(N_WORST * N_FREQ_BINS, color="seagreen", linestyle="--",
               label=f"fully-digital combined ({N_WORST*N_FREQ_BINS} multiplies/update)")
    ax.axvline(min_n_rf_near_ideal, color="gray", linestyle=":",
               label=f"sweet spot (N_RF={min_n_rf_near_ideal})")
    ax.set_xlabel("Number of RF chains (N_RF)")
    ax.set_ylabel("Digital multiplies per beam-update")
    ax.set_title("Stage 14: Digital compute cost vs. N_RF")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/stage14_hybrid.png", dpi=150)
    print("Saved figures/stage14_hybrid.png")

    print("\n=== Stage 14 checkpoint PASSED. Fully-connected hybrid "
          "beamforming validated against both regression limits (N_RF=1 "
          "matches nearfield, N_RF=N matches combined), and the sweet-spot "
          "RF-chain count identified. ===")


if __name__ == "__main__":
    main()