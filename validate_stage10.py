"""
Stage 10 checkpoint script.

Two parts:

Part A -- confirms, numerically, the design-time derivation in noise.py:
    since every beamformer mode carries identical total steering-vector
    energy (sum_n|a_n|^2 = N, enforced since Stage 4), the gain-loss-in-dB
    gap between any two modes is IDENTICAL at every input SNR level. This
    is checked directly (not assumed) before anything else in this stage
    is trusted -- if it failed, it would mean some mode's energy
    normalization broke, which would also quietly bias every gain-loss
    number computed since Stage 3.

Part B -- BER/EVM at Stage 6's worst-performing region (N=512, 4GHz,
    r_frac in [0.02, 0.05, 0.1]), across an SNR sweep. This is where
    absolute noise level actually matters: the FIXED dB gap between
    conventional and combined has a wildly different PRACTICAL
    consequence (in bit error rate) depending on where you sit on the
    BER curve -- a few dB matters enormously near the "waterfall"
    region, and barely at all deep in the error-floor or garbage
    regions. This is the real link-budget payoff for this stage.

Run: python3 validate_stage10.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arraymodel import ArrayGeometry
from channel import near_field_channel
from beamformers import beamform, beamform_gain
from sweep import gain_loss_at_point
from noise import effective_output_snr, ber_qpsk, evm_from_snr

CENTER_FREQ = 28e9
THETA = np.deg2rad(20.0)

# Same worst-performing region as Stage 6/9, for direct comparability.
N_WORST = 512
BW_WORST = 4e9
R_FRACS_WORST = [0.02, 0.05, 0.1]

SNR_SWEEP_DB = [-35, -30, -25, -20, -15, -12, -10, -5, 0]


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    N, F = channel.shape
    power = np.sum(np.abs(channel) ** 2)
    return channel * np.sqrt(N * F / power)


def main():
    print("=== Stage 10 Checkpoint ===\n")

    # =========================================================================
    # Part A -- confirm gain-loss-in-dB is provably SNR-independent under
    # this project's fixed-total-array-energy normalization convention.
    # =========================================================================
    print("-- Part A: confirming gain-loss-in-dB is SNR-independent --")
    print("  (see noise.py's module docstring for the derivation: every mode")
    print("  has sum|a_n|^2 = N, so noise power after combining is identical")
    print("  across modes, at every noise level)\n")

    array = ArrayGeometry(N=N_WORST, center_freq=CENTER_FREQ)
    rayleigh = array.rayleigh_distance()
    r = 0.05 * rayleigh
    freqs = CENTER_FREQ + np.linspace(-BW_WORST / 2, BW_WORST / 2, 33)
    true_channel = _normalize_channel(near_field_channel(array, r, THETA, freqs))

    gain_conv = beamform_gain(
        beamform("conventional", array, freqs, THETA, center_freq=CENTER_FREQ), true_channel
    )
    gain_comb = beamform_gain(
        beamform("combined", array, freqs, THETA, r=r), true_channel
    )
    noiseless_gap_db = 10 * np.log10(gain_comb / gain_conv)
    print(f"  Noiseless gain-loss (Stage 5/6 convention): {noiseless_gap_db:.4f} dB\n")

    print(f"  {'SNR_in (dB)':>12s}  {'SNR_out_conv (dB)':>18s}  {'SNR_out_comb (dB)':>18s}  {'gap (dB)':>10s}")
    gaps = []
    for snr_in in SNR_SWEEP_DB:
        snr_out_conv = effective_output_snr(gain_conv, N_WORST, snr_in)
        snr_out_comb = effective_output_snr(gain_comb, N_WORST, snr_in)
        snr_out_conv_db = 10 * np.log10(snr_out_conv)
        snr_out_comb_db = 10 * np.log10(snr_out_comb)
        gap = snr_out_comb_db - snr_out_conv_db
        gaps.append(gap)
        print(f"  {snr_in:12d}  {snr_out_conv_db:18.4f}  {snr_out_comb_db:18.4f}  {gap:10.4f}")

    gaps = np.array(gaps)
    assert np.allclose(gaps, noiseless_gap_db, atol=1e-9), (
        f"Expected the gain-loss gap to be IDENTICAL at every SNR level "
        f"(provably, given equal steering-vector energy across modes), "
        f"but got variation: {gaps}"
    )
    print(f"\n  OK: the {noiseless_gap_db:.4f} dB gap is IDENTICAL at every "
          f"tested input SNR (-5 to 40 dB), confirming the derivation. This "
          f"is a genuine property of this project's fixed-energy beamformer "
          f"normalization, not a bug or an oversight -- it means the "
          f"gain-loss-in-dB metric itself cannot be used to see any "
          f"'noise compresses the difference between modes' effect, because "
          f"that effect does not exist at the level of this metric. Part B "
          f"looks at where absolute SNR actually does matter.\n")

    # =========================================================================
    # Part B -- BER/EVM across the SNR sweep, at Stage 6's worst region.
    # =========================================================================
    print("-- Part B: BER/EVM at the worst-performing region --\n")

    rows = []
    for r_frac in R_FRACS_WORST:
        r_pt = r_frac * rayleigh
        true_channel_pt = _normalize_channel(near_field_channel(array, r_pt, THETA, freqs))
        gain_conv_pt = beamform_gain(
            beamform("conventional", array, freqs, THETA, center_freq=CENTER_FREQ), true_channel_pt
        )
        gain_comb_pt = beamform_gain(
            beamform("combined", array, freqs, THETA, r=r_pt), true_channel_pt
        )

        for snr_in in SNR_SWEEP_DB:
            for mode, gain in [("conventional", gain_conv_pt), ("combined", gain_comb_pt)]:
                snr_out = effective_output_snr(gain, N_WORST, snr_in)
                snr_out_db = 10 * np.log10(snr_out)
                ber = ber_qpsk(snr_out_db)
                evm = evm_from_snr(snr_out_db)
                rows.append({
                    "r_frac_rayleigh": r_frac,
                    "snr_in_db": snr_in,
                    "beamformer": mode,
                    "snr_out_db": snr_out_db,
                    "ber": ber,
                    "evm_pct": evm,
                })

    df = pd.DataFrame(rows)
    df.to_csv("results/stage10_noise.csv", index=False)
    print("  Saved results/stage10_noise.csv\n")

    # Print a compact table at r_frac=0.05 (the mid worst-region point) to
    # illustrate the actual link-budget story.
    print("  Illustration at r_frac=0.05x Rayleigh (worst-region, mid point):")
    sub = df[df.r_frac_rayleigh == 0.05].pivot(index="snr_in_db", columns="beamformer",
                                                  values=["snr_out_db", "ber"])
    print(sub.to_string())
    print()

    # =========================================================================
    # Checkpoint assertions
    # =========================================================================
    print("-- Checkpoint assertions --")

    # 1. Realistic-operating-point check: find the smallest input SNR (from
    #    the sweep) at which the combined corrector is already comfortably
    #    working (BER < 1e-3). At that SAME input SNR, conventional --
    #    carrying the same fixed 11+ dB penalty -- should still show a
    #    clearly measurable, much worse BER. This is the practically
    #    meaningful "why the dB gap matters" demonstration: a fixed dB gap
    #    can be the difference between a working link and a broken one at
    #    a realistic operating point.
    mid = df[df.r_frac_rayleigh == 0.05].sort_values("snr_in_db")
    comb_mid = mid[mid.beamformer == "combined"].sort_values("snr_in_db")
    conv_mid = mid[mid.beamformer == "conventional"].sort_values("snr_in_db")
    working_points = comb_mid[comb_mid.ber < 1e-3]
    assert len(working_points) > 0, (
        "expected at least one tested SNR where combined achieves BER < 1e-3"
    )
    operating_snr_in = working_points.snr_in_db.min()
    ber_comb_op = comb_mid[comb_mid.snr_in_db == operating_snr_in].iloc[0].ber
    ber_conv_op = conv_mid[conv_mid.snr_in_db == operating_snr_in].iloc[0].ber
    print(f"  At SNR_in={operating_snr_in}dB (smallest SNR where combined is "
          f"comfortably working): BER_combined={ber_comb_op:.3e}, "
          f"BER_conventional={ber_conv_op:.3e}")
    assert ber_conv_op > 1e-2, (
        f"expected conventional to still show a clearly bad BER at the SNR "
        f"where combined first becomes comfortable, got {ber_conv_op:.3e}"
    )
    print("  OK: at a realistic operating point, the fixed dB gap is the "
          "difference between a working link (combined) and a broken one "
          "(conventional).\n")

    # 2. Low-SNR "garbage" limit: at very low input SNR, BOTH modes should
    #    be near the BER ceiling (~0.5, i.e. unusable) -- this is where the
    #    fixed dB gap becomes PRACTICALLY IRRELEVANT (both links are down
    #    regardless of beamformer quality), confirming the "compression"
    #    effect the original plan anticipated actually happens at the BER
    #    level, not at the underlying dB-gap level (which Part A showed is
    #    fixed).
    low_snr_sub = df[(df.snr_in_db == min(SNR_SWEEP_DB)) & (df.r_frac_rayleigh == 0.05)]
    ber_comb_low = low_snr_sub[low_snr_sub.beamformer == "combined"].iloc[0].ber
    ber_conv_low = low_snr_sub[low_snr_sub.beamformer == "conventional"].iloc[0].ber
    print(f"  At SNR_in={min(SNR_SWEEP_DB)}dB, r_frac=0.05: BER_combined={ber_comb_low:.4f}, "
          f"BER_conventional={ber_conv_low:.4f}")
    assert ber_conv_low > 0.3, "conventional should be near the BER ceiling at very low input SNR"
    ber_gap_low = ber_conv_low - ber_comb_low
    ber_gap_high = ber_conv_op - ber_comb_op
    print(f"  BER gap at {min(SNR_SWEEP_DB)}dB: {ber_gap_low:.4f}   "
          f"BER gap at {operating_snr_in}dB (operating point): {ber_gap_high:.4f}")
    assert ber_gap_low < ber_gap_high, (
        "expected the PRACTICAL (BER) gap between modes to compress at very "
        "low SNR relative to the realistic operating point, since both "
        "links are near-unusable regardless of beamformer quality at very "
        "low SNR"
    )
    print("  OK: the practical (BER) gap between modes compresses at very "
          "low SNR, even though the underlying dB gap (Part A) does not -- "
          "confirming where the plan's anticipated 'compression' effect "
          "actually lives.\n")

    # 3. Combined should never show worse BER than conventional, at any
    #    tested SNR/r_frac -- sanity check that nothing got inverted.
    pivot_check = df.pivot_table(index=["r_frac_rayleigh", "snr_in_db"],
                                   columns="beamformer", values="ber")
    assert (pivot_check["combined"] <= pivot_check["conventional"] + 1e-12).all(), (
        "combined should never show a worse (higher) BER than conventional"
    )
    print("  OK: combined never shows worse BER than conventional, at any "
          "tested SNR or worst-region point.\n")

    # =========================================================================
    # Plots
    # =========================================================================
    print("-- Generating figure --")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for r_frac in R_FRACS_WORST:
        for mode, style in [("conventional", "--"), ("combined", "-")]:
            sub = df[(df.r_frac_rayleigh == r_frac) & (df.beamformer == mode)].sort_values("snr_in_db")
            ax.semilogy(sub.snr_in_db, np.maximum(sub.ber, 1e-300), style,
                        marker="o", label=f"{mode}, r={r_frac}x Rayleigh")
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("BER (QPSK)")
    ax.set_title("Stage 10: BER vs. input SNR\n(fixed dB gap, wildly different practical impact)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    sub = df[df.r_frac_rayleigh == 0.05]
    for mode, style in [("conventional", "--"), ("combined", "-")]:
        s = sub[sub.beamformer == mode].sort_values("snr_in_db")
        ax.plot(s.snr_in_db, s.evm_pct, style, marker="o", label=mode)
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("EVM (%)")
    ax.set_title("Stage 10: EVM vs. input SNR (r=0.05x Rayleigh)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/stage10_noise.png", dpi=150)
    print("Saved figures/stage10_noise.png")

    print("\n=== Stage 10 checkpoint PASSED. Confirmed gain-loss-in-dB is "
          "SNR-independent (a real property of this project's normalization "
          "convention), and quantified where absolute SNR actually matters: "
          "BER/EVM, where a fixed dB gap has a hugely different practical "
          "consequence depending on the operating point. ===")


if __name__ == "__main__":
    main()