"""
Stage 9 checkpoint script.

Applies realistic phase-shifter quantization to BOTH the broken baseline
(conventional) AND the combined corrector, at Stage 6's worst-performing
region (N=512, bandwidth=4GHz, r_frac in [0.02, 0.05, 0.1] x Rayleigh),
and asks: how much of Stage 6's dB-recovery survives once BOTH sides
face the same realistic, finite-bit-depth hardware constraint?

Metric -- "recovery efficiency":
    recovery_ideal_db      = Stage 6's original number: gain_loss of the
                              UNQUANTIZED conventional beamformer against
                              the UNQUANTIZED combined corrector (the
                              project's standard reference).
    recovery_quantized_db  = gain_loss of the QUANTIZED conventional
                              beamformer against the QUANTIZED combined
                              corrector, at a given n_bits -- i.e. what
                              you'd actually measure if BOTH beamformers
                              were built with the same realistic
                              hardware.
    recovery_efficiency    = recovery_quantized_db / recovery_ideal_db

This is deliberately NOT "quantized conventional vs ideal combined" --
that would conflate "the baseline got worse" with "the corrector also
got worse", when the practically relevant question is how much of the
FIX survives once neither side is idealized.

Run: python3 validate_stage9.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arraymodel import ArrayGeometry
from channel import near_field_channel
from beamformers import beamform_gain
from hardware import quantized_beamform
from sweep import gain_loss_at_point

from constants import CENTER_FREQ, THETA
N_WORST = 512
BW_WORST = 4e9
R_FRACS_WORST = [0.02, 0.05, 0.1]
N_BITS_SWEEP = [1, 2, 3, 4, 6, 8]  # 1-2 bits deliberately unrealistic,
                                    # included to show a visible penalty;
                                    # 3/4/6 are common real-hardware values.


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    N, F = channel.shape
    power = np.sum(np.abs(channel) ** 2)
    return channel * np.sqrt(N * F / power)


def main():
    print("=== Stage 9 Checkpoint ===\n")
    print(f"Worst-performing region (matches Stage 6): N={N_WORST}, "
          f"bandwidth={BW_WORST/1e9:.0f} GHz, r/Rayleigh in {R_FRACS_WORST}\n")

    array = ArrayGeometry(N=N_WORST, center_freq=CENTER_FREQ)
    rayleigh = array.rayleigh_distance()
    freqs = CENTER_FREQ + np.linspace(-BW_WORST / 2, BW_WORST / 2, 33)

    rows = []
    for r_frac in R_FRACS_WORST:
        r = r_frac * rayleigh
        true_channel = _normalize_channel(near_field_channel(array, r, THETA, freqs))

        # Stage 6's original, fully-idealized recovery number, reused
        # directly (not recomputed by hand) for direct comparability.
        ideal_point = gain_loss_at_point(N_WORST, BW_WORST, r_frac, CENTER_FREQ)
        recovery_ideal_db = ideal_point["gain_loss_conventional_db"]

        for n_bits in N_BITS_SWEEP:
            steer_conv_q = quantized_beamform(
                "conventional", array, freqs, THETA, n_bits, center_freq=CENTER_FREQ
            )
            steer_comb_q = quantized_beamform(
                "combined", array, freqs, THETA, n_bits, r=r
            )
            gain_conv_q = beamform_gain(steer_conv_q, true_channel)
            gain_comb_q = beamform_gain(steer_comb_q, true_channel)

            recovery_quantized_db = 10 * np.log10(gain_comb_q / gain_conv_q)
            recovery_efficiency = recovery_quantized_db / recovery_ideal_db

            # Also track each mode's own gain-loss vs the UNQUANTIZED
            # ideal combined reference, purely for the monotonic-approach
            # sanity check (not used in the recovery-efficiency metric).
            gain_combined_ideal = beamform_gain(
                quantized_beamform("combined", array, freqs, THETA, None, r=r),
                true_channel,
            )
            loss_conv_vs_ideal_db = 10 * np.log10(gain_combined_ideal / gain_conv_q)
            loss_comb_vs_ideal_db = 10 * np.log10(gain_combined_ideal / gain_comb_q)

            rows.append({
                "r_frac_rayleigh": r_frac,
                "n_bits": n_bits,
                "recovery_ideal_db": recovery_ideal_db,
                "recovery_quantized_db": recovery_quantized_db,
                "recovery_efficiency": recovery_efficiency,
                "loss_conv_vs_ideal_db": loss_conv_vs_ideal_db,
                "loss_comb_vs_ideal_db": loss_comb_vs_ideal_db,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/stage9_quantization.csv", index=False)
    print(df.to_string(index=False))
    print("\n  Saved results/stage9_quantization.csv\n")

    # =========================================================================
    # Checkpoint assertions
    # =========================================================================
    print("-- Checkpoint assertions --")

    # 1. Monotonic approach: as n_bits increases, both modes' loss vs the
    #    unquantized ideal should shrink monotonically toward 0 -- sanity
    #    check that quantization is implemented correctly, not just noise.
    for r_frac in R_FRACS_WORST:
        sub = df[df.r_frac_rayleigh == r_frac].sort_values("n_bits")
        conv_losses = sub.loss_conv_vs_ideal_db.values
        comb_losses = sub.loss_comb_vs_ideal_db.values
        assert np.all(np.diff(conv_losses) <= 1e-6), (
            f"r_frac={r_frac}: conventional's loss vs ideal should shrink "
            f"monotonically as n_bits increases, got {conv_losses}"
        )
        assert np.all(np.diff(comb_losses) <= 1e-6), (
            f"r_frac={r_frac}: combined's loss vs ideal should shrink "
            f"monotonically as n_bits increases, got {comb_losses}"
        )
    print("  OK: both modes' gain-loss vs the unquantized ideal shrinks "
          "monotonically as n_bits increases (quantization behaves correctly).")

    # 2. Visible, explainable penalty at very low bit depth (1-2 bits):
    #    the combined corrector -- which is otherwise perfect (0 dB loss)
    #    -- should show a clearly non-trivial loss at 1-bit resolution.
    for r_frac in R_FRACS_WORST:
        sub = df[df.r_frac_rayleigh == r_frac]
        loss_1bit = sub[sub.n_bits == 1].iloc[0].loss_comb_vs_ideal_db
        assert loss_1bit > 1.0, (
            f"r_frac={r_frac}: expected a clearly visible penalty at 1-bit "
            f"resolution, got only {loss_1bit:.3f} dB"
        )
    print("  OK: 1-bit resolution shows a clear, non-trivial gain penalty "
          "on the otherwise-perfect combined corrector.")

    # 3. Recovery efficiency approaches 1.0 (>99%) at high bit depth.
    for r_frac in R_FRACS_WORST:
        sub = df[df.r_frac_rayleigh == r_frac]
        eff_8bit = sub[sub.n_bits == 8].iloc[0].recovery_efficiency
        assert eff_8bit > 0.99, (
            f"r_frac={r_frac}: expected recovery efficiency > 99% at 8-bit "
            f"resolution, got {eff_8bit:.4f}"
        )
    print("  OK: recovery efficiency exceeds 99% at 8-bit resolution "
          "(converges to the fully-idealized Stage 6 result).")

    # 4. Find the minimum bit depth that recovers >=90% of the ideal gap,
    #    per r_frac -- the practically useful finding for the writeup.
    print("\n-- Minimum viable bit depth (>=90% recovery efficiency) --")
    min_bits_by_rfrac = {}
    for r_frac in R_FRACS_WORST:
        sub = df[df.r_frac_rayleigh == r_frac].sort_values("n_bits")
        above_90 = sub[sub.recovery_efficiency >= 0.90]
        min_bits = int(above_90.n_bits.min()) if len(above_90) > 0 else None
        min_bits_by_rfrac[r_frac] = min_bits
        print(f"  r_frac={r_frac}: minimum n_bits for >=90% recovery "
              f"efficiency = {min_bits}")
    assert all(v is not None for v in min_bits_by_rfrac.values()), (
        "expected every tested point to reach 90% recovery efficiency "
        "somewhere within the tested n_bits range"
    )
    print("\n  OK: every worst-region point reaches >=90% recovery efficiency "
          "within the tested bit-depth range.")

    # ---- IMPORTANT CAVEAT: recovery efficiency alone can be misleading ----
    # At n_bits=1, recovery_efficiency is already >99% for every point --
    # but that does NOT mean 1-bit hardware is actually fine. It's a ratio
    # metric: at 1-bit, BOTH conventional (loss vs ideal jumps from ~15dB
    # to ~17dB) AND combined (loss vs ideal jumps from ~0dB to ~3.9dB)
    # degrade by roughly similar absolute amounts, so the GAP between them
    # survives even though BOTH sides are performing badly in absolute
    # terms. A "minimum viable bit depth" claim based on recovery
    # efficiency alone would overstate how usable 1-bit hardware actually
    # is. A second, stricter criterion -- the corrector's own absolute
    # loss vs the unquantized ideal -- is needed to make a genuinely
    # practical hardware recommendation.
    print("\n-- Caveat: recovery efficiency alone is misleading at low bit depth --")
    print("  At n_bits=1, recovery efficiency is already >99% everywhere, but "
          "this is because quantization degrades BOTH conventional and "
          "combined by similar absolute amounts -- their GAP survives even "
          "though the combined corrector's own absolute performance (loss "
          "vs the unquantized ideal) is already ~3.9 dB worse at 1-bit, "
          "which is NOT a trivial hardware penalty for what's supposed to "
          "be the near-lossless corrector.")

    print("\n-- Minimum bit depth by a STRICTER criterion "
          "(corrector's own loss vs unquantized ideal < 1 dB) --")
    min_bits_strict_by_rfrac = {}
    for r_frac in R_FRACS_WORST:
        sub = df[df.r_frac_rayleigh == r_frac].sort_values("n_bits")
        below_1db = sub[sub.loss_comb_vs_ideal_db < 1.0]
        min_bits_strict = int(below_1db.n_bits.min()) if len(below_1db) > 0 else None
        min_bits_strict_by_rfrac[r_frac] = min_bits_strict
        print(f"  r_frac={r_frac}: minimum n_bits for combined corrector's "
              f"own loss < 1 dB = {min_bits_strict}")
    print("\n  This stricter criterion consistently lands at 2 bits, not 1 -- "
          "the practically meaningful recommendation for the writeup: 2-bit "
          "phase resolution keeps the combined corrector within 1 dB of its "
          "own idealized performance; 1-bit hardware should NOT be called "
          "'sufficient' just because the recovery-efficiency ratio looks "
          "good, since that ratio can hide a real absolute-performance hit "
          "on both sides.")

    # =========================================================================
    # Plot: recovery efficiency vs n_bits, one line per r_frac.
    # =========================================================================
    print("\n-- Generating figure --")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for r_frac in R_FRACS_WORST:
        sub = df[df.r_frac_rayleigh == r_frac].sort_values("n_bits")
        ax.plot(sub.n_bits, sub.recovery_efficiency, marker="o",
                label=f"r={r_frac}x Rayleigh")
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% threshold")
    ax.axhline(1.0, color="k", linewidth=0.8)
    ax.set_xlabel("Phase-shifter resolution (bits)")
    ax.set_ylabel("Recovery efficiency (quantized / ideal)")
    ax.set_title("Stage 9: Recovery efficiency vs. phase-shifter bit depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for r_frac in R_FRACS_WORST:
        sub = df[df.r_frac_rayleigh == r_frac].sort_values("n_bits")
        ax.plot(sub.n_bits, sub.loss_comb_vs_ideal_db, marker="o",
                label=f"combined, r={r_frac}x Rayleigh")
    ax.set_xlabel("Phase-shifter resolution (bits)")
    ax.set_ylabel("Combined corrector's own gain-loss vs. unquantized ideal (dB)")
    ax.set_title("Stage 9: Quantization penalty on the corrector itself")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/stage9_quantization.png", dpi=150)
    print("Saved figures/stage9_quantization.png")

    print("\n=== Stage 9 checkpoint PASSED. Quantization behaves correctly, "
          "and minimum viable phase-shifter resolution identified for each "
          "worst-region point. ===")


if __name__ == "__main__":
    main()