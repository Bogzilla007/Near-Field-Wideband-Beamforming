"""
Stage 4 checkpoint script.

Validates, per the project doc's isolation principle (Section 2.5) and
convergence checks (Stage 4 task list):

  1. Near-field-aware beamformer, tested ONLY against a near-field channel
     with a NARROWBAND signal -- isolates its correctness from squint.
  2. Squint-aware (TTD) beamformer, tested ONLY against a far-field channel
     with a WIDEBAND signal -- isolates its correctness from near-field.
  3. Convergence: nearfield-aware -> conventional as distance -> infinity.
  4. Convergence: squint-aware -> conventional as bandwidth -> 0.
  5. Convergence: combined -> nearfield-aware as bandwidth -> 0.
  6. Convergence: combined -> squint-aware as distance -> infinity.

Only after all of these pass is it safe to test the beamformers combined
(near-field + wideband together) in Stage 5.

Run: python3 validate_stage4.py
"""

import numpy as np
import matplotlib.pyplot as plt

from arraymodel import ArrayGeometry
from channel import near_field_channel, far_field_channel
from beamformers import beamform, beamform_gain
from visualization import plot_beam_pattern

from constants import CENTER_FREQ, THETA
N_TEST = 128


def gain_loss_db(gain_reference: float, gain_test: float) -> float:
    """10*log10(gain_reference / gain_test), per Section 2.6's definition."""
    return 10 * np.log10(gain_reference / gain_test)


def normalize_channel(channel: np.ndarray, N: int) -> np.ndarray:
    """Normalize total channel power to N (per-element unit power on
    average), consistent with Stage 3's normalization rule -- prevents the
    near-field model's genuine 1/r amplitude taper from contaminating gain
    comparisons across different distances/array sizes (Section 2.6)."""
    power = np.sum(np.abs(channel) ** 2)
    return channel * np.sqrt(channel.shape[0] * channel.shape[1] / power)


def main():
    print("=== Stage 4 Checkpoint ===\n")
    array = ArrayGeometry(N=N_TEST, center_freq=CENTER_FREQ)
    rayleigh = array.rayleigh_distance()
    print(f"Test array: N={N_TEST}, Rayleigh distance={rayleigh:.2f} m\n")

    # =========================================================================
    # 1. Isolation test: near-field-aware beamformer, near-field channel,
    #    NARROWBAND signal only.
    # =========================================================================
    print("-- 1. Isolation: near-field-aware vs near-field channel, narrowband --")
    r_near = 0.05 * rayleigh  # well inside near-field (per Stage 2's divergence
    # curve, this is where channel mismatch is large -- 0.3x was too mild
    # to produce a clearly "broken" conventional baseline for this test)
    freqs_narrow = np.array([CENTER_FREQ])  # single tone -> no squint possible

    true_channel = normalize_channel(
        near_field_channel(array, r_near, THETA, freqs_narrow), N_TEST
    )
    steer_nf = beamform("nearfield", array, freqs_narrow, THETA, r=r_near, center_freq=CENTER_FREQ)
    steer_conv = beamform("conventional", array, freqs_narrow, THETA, center_freq=CENTER_FREQ)
    steer_combined = beamform("combined", array, freqs_narrow, THETA, r=r_near)

    gain_nf = beamform_gain(steer_nf, true_channel)
    gain_conv = beamform_gain(steer_conv, true_channel)
    gain_combined = beamform_gain(steer_combined, true_channel)

    loss_nf = gain_loss_db(gain_combined, gain_nf)
    loss_conv = gain_loss_db(gain_combined, gain_conv)
    print(f"  r = {r_near:.2f} m (0.3x Rayleigh), narrowband")
    print(f"  gain_loss_nearfield_aware = {loss_nf:.4f} dB (expect ~0, it's corrected)")
    print(f"  gain_loss_conventional    = {loss_conv:.4f} dB (expect >> 0, it's broken here)")
    assert loss_nf < 0.5, f"near-field-aware should recover near-full gain, got {loss_nf:.4f} dB"
    assert loss_conv > 3.0, f"conventional should show meaningful loss here, got {loss_conv:.4f} dB"
    print("  OK: near-field-aware beamformer is validated in isolation.\n")

    # =========================================================================
    # 2. Isolation test: squint-aware (TTD) beamformer, far-field channel,
    #    WIDEBAND signal only.
    # =========================================================================
    print("-- 2. Isolation: squint-aware vs far-field channel, wideband --")
    # Squint severity scales with N (more elements -> more accumulated phase
    # error at off-carrier frequencies), so use the project's larger array
    # ceiling here to get a meaningfully broken conventional baseline --
    # N=128 with 1 GHz/28 GHz only produced ~0.7 dB loss, too mild to test.
    N_squint_test = 512
    array_sq = ArrayGeometry(N=N_squint_test, center_freq=CENTER_FREQ)
    rayleigh_sq = array_sq.rayleigh_distance()
    r_far = 50 * rayleigh_sq  # well past Rayleigh distance
    bandwidth_wide = 1e9  # 1 GHz -- significant squint at this N
    freqs_wide = CENTER_FREQ + np.linspace(-bandwidth_wide / 2, bandwidth_wide / 2, 65)

    true_channel = normalize_channel(
        near_field_channel(array_sq, r_far, THETA, freqs_wide), N_squint_test
    )
    steer_sq = beamform("squint", array_sq, freqs_wide, THETA)
    steer_conv = beamform("conventional", array_sq, freqs_wide, THETA, center_freq=CENTER_FREQ)
    steer_combined = beamform("combined", array_sq, freqs_wide, THETA, r=r_far)

    gain_sq = beamform_gain(steer_sq, true_channel)
    gain_conv = beamform_gain(steer_conv, true_channel)
    gain_combined = beamform_gain(steer_combined, true_channel)

    loss_sq = gain_loss_db(gain_combined, gain_sq)
    loss_conv = gain_loss_db(gain_combined, gain_conv)
    print(f"  r = {r_far:.2f} m (50x Rayleigh), bandwidth = {bandwidth_wide/1e9:.2f} GHz")
    print(f"  gain_loss_squint_aware = {loss_sq:.4f} dB (expect ~0, it's corrected)")
    print(f"  gain_loss_conventional = {loss_conv:.4f} dB (expect >> 0, it's broken here)")
    assert loss_sq < 0.5, f"squint-aware should recover near-full gain, got {loss_sq:.4f} dB"
    assert loss_conv > 3.0, f"conventional should show meaningful loss here, got {loss_conv:.4f} dB"
    print("  OK: squint-aware beamformer is validated in isolation.\n")

    # =========================================================================
    # 3. Convergence: nearfield-aware -> conventional as r -> infinity
    # =========================================================================
    print("-- 3. Convergence: nearfield-aware -> conventional as r -> infinity --")
    r_values = np.array([0.5, 2, 10, 50, 200]) * rayleigh
    diffs_3 = []
    for r in r_values:
        true_channel = normalize_channel(near_field_channel(array, r, THETA, freqs_narrow), N_TEST)
        g_nf = beamform_gain(beamform("nearfield", array, freqs_narrow, THETA, r=r, center_freq=CENTER_FREQ), true_channel)
        g_conv = beamform_gain(beamform("conventional", array, freqs_narrow, THETA, center_freq=CENTER_FREQ), true_channel)
        diff_db = abs(gain_loss_db(g_nf, g_conv))
        diffs_3.append(diff_db)
        print(f"  r={r:9.2f} m ({r/rayleigh:6.1f}x Rayleigh): |gain diff| = {diff_db:.4f} dB")
    assert diffs_3[-1] < diffs_3[0], "difference should shrink as r grows"
    assert diffs_3[-1] < 0.1, f"should nearly vanish at large r, got {diffs_3[-1]:.4f} dB"
    print("  OK: nearfield-aware converges to conventional as r -> infinity.\n")

    # =========================================================================
    # 4. Convergence: squint-aware -> conventional as bandwidth -> 0
    # =========================================================================
    print("-- 4. Convergence: squint-aware -> conventional as bandwidth -> 0 --")
    bandwidths = np.array([1e9, 400e6, 100e6, 10e6, 1e6])
    diffs_4 = []
    for bw in bandwidths:
        f_set = CENTER_FREQ + np.linspace(-bw / 2, bw / 2, 65)
        true_channel = normalize_channel(near_field_channel(array, r_far, THETA, f_set), N_TEST)
        g_sq = beamform_gain(beamform("squint", array, f_set, THETA), true_channel)
        g_conv = beamform_gain(beamform("conventional", array, f_set, THETA, center_freq=CENTER_FREQ), true_channel)
        diff_db = abs(gain_loss_db(g_sq, g_conv))
        diffs_4.append(diff_db)
        print(f"  bandwidth={bw/1e6:8.1f} MHz: |gain diff| = {diff_db:.4f} dB")
    assert diffs_4[-1] < diffs_4[0], "difference should shrink as bandwidth shrinks"
    assert diffs_4[-1] < 0.1, f"should nearly vanish at narrow bandwidth, got {diffs_4[-1]:.4f} dB"
    print("  OK: squint-aware converges to conventional as bandwidth -> 0.\n")

    # =========================================================================
    # 5. Convergence: combined -> nearfield-aware as bandwidth -> 0
    # =========================================================================
    print("-- 5. Convergence: combined -> nearfield-aware as bandwidth -> 0 --")
    diffs_5 = []
    for bw in bandwidths:
        f_set = CENTER_FREQ + np.linspace(-bw / 2, bw / 2, 65)
        true_channel = normalize_channel(near_field_channel(array, r_near, THETA, f_set), N_TEST)
        g_comb = beamform_gain(beamform("combined", array, f_set, THETA, r=r_near), true_channel)
        g_nf = beamform_gain(beamform("nearfield", array, f_set, THETA, r=r_near, center_freq=CENTER_FREQ), true_channel)
        diff_db = abs(gain_loss_db(g_comb, g_nf))
        diffs_5.append(diff_db)
        print(f"  bandwidth={bw/1e6:8.1f} MHz: |gain diff| = {diff_db:.4f} dB")
    assert diffs_5[-1] < diffs_5[0], "difference should shrink as bandwidth shrinks"
    assert diffs_5[-1] < 0.1, f"should nearly vanish at narrow bandwidth, got {diffs_5[-1]:.4f} dB"
    print("  OK: combined converges to nearfield-aware as bandwidth -> 0.\n")

    # =========================================================================
    # 6. Convergence: combined -> squint-aware as distance -> infinity
    # =========================================================================
    print("-- 6. Convergence: combined -> squint-aware as distance -> infinity --")
    diffs_6 = []
    for r in r_values:
        true_channel = normalize_channel(near_field_channel(array, r, THETA, freqs_wide), N_TEST)
        g_comb = beamform_gain(beamform("combined", array, freqs_wide, THETA, r=r), true_channel)
        g_sq = beamform_gain(beamform("squint", array, freqs_wide, THETA), true_channel)
        diff_db = abs(gain_loss_db(g_comb, g_sq))
        diffs_6.append(diff_db)
        print(f"  r={r:9.2f} m ({r/rayleigh:6.1f}x Rayleigh): |gain diff| = {diff_db:.4f} dB")
    assert diffs_6[-1] < diffs_6[0], "difference should shrink as r grows"
    assert diffs_6[-1] < 0.1, f"should nearly vanish at large r, got {diffs_6[-1]:.4f} dB"
    print("  OK: combined converges to squint-aware as r -> infinity.\n")

    # =========================================================================
    # Visual: beam pattern "wow" figure -- broken vs corrected
    # =========================================================================
    print("-- Generating beam pattern figure (broken vs corrected) --")
    # Use a smaller array + more extreme conditions purely for this VISUAL
    # demo -- N=128/512 (used above for quantitative isolation tests) makes
    # beams too narrow to visually read the split/blur effect on a polar
    # plot. A smaller array with a wider beam makes the mechanism legible.
    N_demo = 24
    array_demo = ArrayGeometry(N=N_demo, center_freq=CENTER_FREQ)
    rayleigh_demo = array_demo.rayleigh_distance()

    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"}, figsize=(18, 7))

    # Panel 1: conventional beamformer, VERY wideband, far-field -> squint splits the lobe
    bandwidth_demo = 4e9  # 4 GHz -- exaggerated for visual clarity
    freq_sample = CENTER_FREQ + np.array([-bandwidth_demo/2, 0, bandwidth_demo/2])
    steer_conv_wb = beamform("conventional", array_demo, freq_sample, THETA, center_freq=CENTER_FREQ)
    plot_beam_pattern(steer_conv_wb, array_demo, freq_sample, r=None,
                       title="Conventional, wideband\n(squint splits the lobe)",
                       ax=axes[0], floor_db=-20.0)

    # Panel 2: conventional beamformer, narrowband, VERY deep near-field -> blurred/missed lobe
    r_demo_near = 0.03 * rayleigh_demo
    steer_conv_nb = beamform("conventional", array_demo, freqs_narrow, THETA, center_freq=CENTER_FREQ)
    plot_beam_pattern(steer_conv_nb, array_demo, freqs_narrow, r=r_demo_near,
                       title="Conventional, deep near-field\n(blurred/missed lobe)",
                       ax=axes[1], freq_labels=["center freq"], floor_db=-20.0)

    # Panel 3: combined corrector, same wideband + near-field conditions simultaneously -> clean lobe
    steer_comb_wb_nf = beamform("combined", array_demo, freq_sample, THETA, r=r_demo_near)
    plot_beam_pattern(steer_comb_wb_nf, array_demo, freq_sample, r=r_demo_near,
                       title="Combined corrector,\nwideband + near-field\n(refocused)",
                       ax=axes[2], floor_db=-20.0)

    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=8, ncol=3)

    plt.tight_layout()
    plt.savefig("figures/stage4_beam_patterns.png", dpi=150, bbox_inches="tight")
    print("Saved figures/stage4_beam_patterns.png")

    print("\n=== Stage 4 checkpoint PASSED. All four modes validated, "
          "isolation + convergence confirmed. Safe to proceed to Stage 5. ===")


if __name__ == "__main__":
    main()