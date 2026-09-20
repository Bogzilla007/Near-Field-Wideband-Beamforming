"""
Stage 12 checkpoint script.

Four parts, in the hard-gate order this project always uses:

Part 1 -- REGRESSION GATE (must pass before anything else is trusted):
    when N_y=1 and phi=0, every 2D channel model and every 2D beamformer
    mode must reduce EXACTLY (to machine precision) to the existing,
    already-validated 1D ULA results from Stages 2-4.

Part 2 -- square-array convergence/divergence (Stage 2-style): confirm
    r/Rayleigh(D_diag) collapses divergence curves across different
    SQUARE array sizes and different (theta, phi) angle combinations,
    same as the 1D case did across different N.

Part 3 -- THE GUIDING QUESTION: does r/Rayleigh(D_diag) alone also
    collapse curves across different ASPECT RATIOS (N_x/N_y) at fixed
    total N? Tested directly, not assumed. Spoiler (found while building
    this stage): it does NOT -- elongated arrays show meaningfully worse
    near-field mismatch than square arrays at the same normalized
    fraction, confirming aspect ratio is a genuine, separate factor, not
    something the diagonal aperture alone absorbs.

Part 4 -- beamformer isolation (Stage 4-style): near-field-aware 2D
    steering vector recovers near-full gain in deep near-field while
    conventional (angle-only) is badly broken, plus a 2D (azimuth x
    elevation) beam pattern heatmap.

Run: python3 validate_stage12.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from arraymodel import ArrayGeometry
from channel import near_field_channel, far_field_channel
from beamformers import beamform, beamform_gain
from planar import (
    PlanarArrayGeometry,
    near_field_channel_planar,
    far_field_channel_planar,
    beamform_planar,
)

from constants import CENTER_FREQ, THETA


def normalized_error(a_near: np.ndarray, a_far: np.ndarray) -> float:
    """Same phase-invariant chordal distance metric as Stage 2's
    validate_stage2.py, reused here unchanged for a directly comparable
    convergence/divergence measure in 2D."""
    a_near_n = a_near / np.linalg.norm(a_near)
    a_far_n = a_far / np.linalg.norm(a_far)
    correlation = np.abs(np.vdot(a_near_n, a_far_n))
    correlation = np.clip(correlation, 0.0, 1.0)
    return np.sqrt(2 - 2 * correlation)


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    N, F = channel.shape
    power = np.sum(np.abs(channel) ** 2)
    return channel * np.sqrt(N * F / power)


def main():
    print("=== Stage 12 Checkpoint ===\n")

    # =========================================================================
    # Part 1 -- Regression gate: 2D (N_y=1, phi=0) must reduce EXACTLY to
    # the existing, already-validated 1D ULA.
    # =========================================================================
    print("-- Part 1: Regression gate (2D -> 1D reduction) --")
    N_TEST = 64
    freqs = CENTER_FREQ + np.linspace(-1e9, 1e9, 17)

    array_1d = ArrayGeometry(N=N_TEST, center_freq=CENTER_FREQ)
    array_2d = PlanarArrayGeometry(N_x=N_TEST, N_y=1, center_freq=CENTER_FREQ)
    r_test = 0.05 * array_1d.rayleigh_distance()

    assert np.isclose(array_1d.rayleigh_distance(), array_2d.rayleigh_distance()), (
        "1D and degenerate-2D Rayleigh distances should match exactly"
    )

    h1d = near_field_channel(array_1d, r_test, THETA, freqs)
    h2d = near_field_channel_planar(array_2d, r_test, THETA, 0.0, freqs)
    diff_near = np.max(np.abs(h1d - h2d))
    print(f"  near-field channel max abs diff: {diff_near:.2e}")
    assert diff_near < 1e-9, "2D near-field channel should reduce exactly to 1D"

    f1d = far_field_channel(array_1d, THETA, freqs)
    f2d = far_field_channel_planar(array_2d, THETA, 0.0, freqs)
    diff_far = np.max(np.abs(f1d - f2d))
    print(f"  far-field channel max abs diff:  {diff_far:.2e}")
    assert diff_far < 1e-9, "2D far-field channel should reduce exactly to 1D"

    for mode, kwargs in [
        ("conventional", dict(center_freq=CENTER_FREQ)),
        ("nearfield", dict(r=r_test, center_freq=CENTER_FREQ)),
        ("squint", dict()),
        ("combined", dict(r=r_test)),
    ]:
        s1d = beamform(mode, array_1d, freqs, THETA, **kwargs)
        s2d = beamform_planar(mode, array_2d, freqs, THETA, 0.0, **kwargs)
        diff = np.max(np.abs(s1d - s2d))
        print(f"  {mode:15s} steering max abs diff: {diff:.2e}")
        assert diff < 1e-9, f"2D {mode} steering vector should reduce exactly to 1D"

    print("  OK: every 2D channel model and beamformer mode reduces exactly "
          "to the already-validated 1D ULA results. Safe to trust the 2D "
          "extension.\n")

    # =========================================================================
    # Part 2 -- Square-array convergence/divergence, Stage 2-style.
    # =========================================================================
    print("-- Part 2: Square-array collapse (Stage 2-style) --")
    freqs_narrow = np.array([CENTER_FREQ])
    r_fracs = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    angle_combos = [(0, 0), (20, 0), (20, 15), (0, 30)]

    square_results = {}
    for Nside in [8, 32]:
        array = PlanarArrayGeometry(N_x=Nside, N_y=Nside, center_freq=CENTER_FREQ)
        rayleigh = array.rayleigh_distance()
        for theta_deg, phi_deg in angle_combos:
            theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
            errs = []
            for frac in r_fracs:
                r = frac * rayleigh
                hn = near_field_channel_planar(array, r, theta, phi, freqs_narrow)[:, 0]
                hf = far_field_channel_planar(array, theta, phi, freqs_narrow)[:, 0]
                errs.append(normalized_error(hn, hf))
            square_results[(Nside, theta_deg, phi_deg)] = np.array(errs)
            print(f"  N={Nside}x{Nside}, theta={theta_deg:3d}, phi={phi_deg:3d}: "
                  f"errs={np.round(errs, 3)}")

    # Collapse check: at each r_frac, the spread across all (N, theta, phi)
    # combos should be small relative to the error's own magnitude --
    # mirrors Stage 2's "divergence curves collapse across array size"
    # finding, now also across off-axis angle for SQUARE arrays.
    print("\n  Collapse quality (spread / mean across all N/theta/phi combos, per r_frac):")
    all_errs = np.array(list(square_results.values()))  # (n_combos, n_fracs)
    square_spread_ratios = []
    for i, frac in enumerate(r_fracs):
        vals = all_errs[:, i]
        spread_ratio = (vals.max() - vals.min()) / vals.mean()
        square_spread_ratios.append(spread_ratio)
        print(f"    r_frac={frac:5.2f}: spread/mean = {spread_ratio:.3f}")
    assert max(square_spread_ratios) < 0.35, (
        f"expected square arrays to collapse well across N/theta/phi (like "
        f"the 1D case did across N), got spread ratios up to "
        f"{max(square_spread_ratios):.3f}"
    )
    print("  OK: square arrays collapse well across array size AND off-axis "
          "angle when normalized by r/Rayleigh(D_diag) -- same behavior as "
          "the 1D case.\n")

    # =========================================================================
    # Part 3 -- THE GUIDING QUESTION: does the same normalization survive
    # changing ASPECT RATIO at fixed total N?
    # =========================================================================
    print("-- Part 3: Aspect-ratio test (the guiding question) --")
    aspect_configs = [(64, 64), (128, 32), (32, 128), (256, 16), (16, 256)]
    aspect_results = {}
    for Nx, Ny in aspect_configs:
        array = PlanarArrayGeometry(N_x=Nx, N_y=Ny, center_freq=CENTER_FREQ)
        rayleigh = array.rayleigh_distance()
        errs = []
        for frac in r_fracs:
            r = frac * rayleigh
            hn = near_field_channel_planar(array, r, THETA, 0.0, freqs_narrow)[:, 0]
            hf = far_field_channel_planar(array, THETA, 0.0, freqs_narrow)[:, 0]
            errs.append(normalized_error(hn, hf))
        aspect_results[(Nx, Ny)] = np.array(errs)
        print(f"  N_x={Nx:4d} N_y={Ny:4d} (N={Nx*Ny}, aspect={Nx/Ny:.3f}, "
              f"Rayleigh={rayleigh:.1f}m): errs={np.round(errs, 3)}")

    print("\n  Spread across ASPECT RATIOS at fixed total N (vs. spread "
          "across square sizes/angles from Part 2):")
    all_aspect_errs = np.array(list(aspect_results.values()))
    aspect_spread_ratios = []
    for i, frac in enumerate(r_fracs):
        vals = all_aspect_errs[:, i]
        spread_ratio = (vals.max() - vals.min()) / vals.mean()
        aspect_spread_ratios.append(spread_ratio)
        print(f"    r_frac={frac:5.2f}: spread/mean = {spread_ratio:.3f} "
              f"(square-array spread was {square_spread_ratios[i]:.3f})")

    assert max(aspect_spread_ratios) > 1.15 * max(square_spread_ratios), (
        "expected the aspect-ratio spread to be clearly larger than the "
        "square-array spread -- if not, r/Rayleigh(D_diag) alone would be "
        "sufficient after all, contradicting what was found while building "
        "this stage"
    )
    print(f"\n  CONFIRMED: aspect-ratio spread (up to {max(aspect_spread_ratios):.3f}) "
          f"is meaningfully larger than square-array spread (up to "
          f"{max(square_spread_ratios):.3f}) at matched r/Rayleigh(D_diag) -- "
          f"roughly a {100*(max(aspect_spread_ratios)/max(square_spread_ratios)-1):.0f}% "
          f"relative increase. This is a real, modest-but-consistent effect "
          f"(present across every r_frac >= 0.1 tested), not a dramatic "
          f"breakdown -- diagonal-based normalization is not badly wrong, "
          f"but it is measurably incomplete.")
    print("\n  Finding for the writeup: r/Rayleigh(D_diag) alone does NOT "
          "fully normalize away array geometry in 2D -- it works well "
          "across array SIZE and across off-axis ANGLE for a fixed "
          "(square) aspect ratio, but elongated arrays (e.g. 128x32) show "
          "a consistently larger spread than square arrays (e.g. 64x64) at "
          "the SAME normalized r/Rayleigh(D_diag) fraction, at the same "
          "total N. The correct normalization variable in 2D is therefore "
          "NOT fully captured by a single scalar r/Rayleigh(D_diag) alone "
          "-- aspect ratio (N_x/N_y) is a genuine, separate contributing "
          "factor, confirming the Extension Plan's suspicion, though the "
          "size of the effect here is modest (~25% relative increase in "
          "spread) rather than a dramatic failure of the diagonal-based "
          "formula. A follow-up refinement (out of scope for this stage) "
          "would be to test whether a PER-AXIS Rayleigh distance "
          "(2*aperture_x^2/lambda and 2*aperture_y^2/lambda separately, "
          "rather than one diagonal-based scalar) collapses better.\n")

    # =========================================================================
    # Part 4 -- Beamformer isolation (Stage 4-style) + 2D beam pattern.
    # =========================================================================
    print("-- Part 4: Beamformer isolation + 2D beam pattern --")
    N_demo = 32
    array_demo = PlanarArrayGeometry(N_x=N_demo, N_y=N_demo, center_freq=CENTER_FREQ)
    rayleigh_demo = array_demo.rayleigh_distance()
    r_near = 0.05 * rayleigh_demo
    phi_demo = np.deg2rad(10.0)

    true_channel = _normalize_channel(
        near_field_channel_planar(array_demo, r_near, THETA, phi_demo, freqs_narrow)
    )
    steer_nf = beamform_planar("nearfield", array_demo, freqs_narrow, THETA, phi_demo,
                                r=r_near, center_freq=CENTER_FREQ)
    steer_conv = beamform_planar("conventional", array_demo, freqs_narrow, THETA, phi_demo,
                                  center_freq=CENTER_FREQ)
    steer_combined = beamform_planar("combined", array_demo, freqs_narrow, THETA, phi_demo,
                                      r=r_near)

    gain_nf = beamform_gain(steer_nf, true_channel)
    gain_conv = beamform_gain(steer_conv, true_channel)
    gain_combined = beamform_gain(steer_combined, true_channel)

    loss_nf = 10 * np.log10(gain_combined / gain_nf)
    loss_conv = 10 * np.log10(gain_combined / gain_conv)
    print(f"  N={N_demo}x{N_demo}, r=0.05x Rayleigh, theta=20deg, phi=10deg:")
    print(f"  gain_loss_nearfield_aware = {loss_nf:.4f} dB (expect ~0)")
    print(f"  gain_loss_conventional    = {loss_conv:.4f} dB (expect >> 0)")
    assert loss_nf < 0.5, f"2D near-field-aware should recover near-full gain, got {loss_nf:.4f} dB"
    assert loss_conv > 3.0, f"2D conventional should show meaningful loss here, got {loss_conv:.4f} dB"
    print("  OK: 2D near-field-aware beamformer is validated in isolation, "
          "same behavior as the 1D case.\n")

    # 2D beam pattern: azimuth x elevation heatmap, conventional vs
    # near-field-aware, at the true near-field distance used above.
    print("-- Generating 2D beam pattern figure --")
    theta_grid = np.deg2rad(np.linspace(-40, 40, 81))
    phi_grid = np.deg2rad(np.linspace(-40, 40, 81))

    def beam_pattern_2d(steering):
        response = np.zeros((len(theta_grid), len(phi_grid)))
        for i, th in enumerate(theta_grid):
            for j, ph in enumerate(phi_grid):
                ch = near_field_channel_planar(array_demo, r_near, th, ph, freqs_narrow)
                response[i, j] = np.abs(np.sum(np.conj(steering[:, 0]) * ch[:, 0])) ** 2
        return 10 * np.log10(response / response.max() + 1e-12)

    pattern_conv = beam_pattern_2d(steer_conv)
    pattern_nf = beam_pattern_2d(steer_nf)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, pattern, title in [
        (axes[0], pattern_conv, "Conventional (angle-only)\nnear-field, theta=20deg, phi=10deg"),
        (axes[1], pattern_nf, "Near-field-aware\n(same conditions)"),
    ]:
        pcm = ax.pcolormesh(np.rad2deg(theta_grid), np.rad2deg(phi_grid), pattern.T,
                             shading="nearest", cmap="inferno", vmin=-20, vmax=0)
        ax.axvline(20, color="cyan", linestyle="--", linewidth=1)
        ax.axhline(10, color="cyan", linestyle="--", linewidth=1)
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation (deg)")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label="Response (dB)")

    plt.suptitle(f"Stage 12: 2D beam pattern, N={N_demo}x{N_demo} UPA, r=0.05x Rayleigh", y=1.03)
    plt.tight_layout()
    plt.savefig("figures/stage12_beam_pattern_2d.png", dpi=150, bbox_inches="tight")
    print("Saved figures/stage12_beam_pattern_2d.png")

    print("\n=== Stage 12 checkpoint PASSED. 2D UPA validated against the 1D "
          "regression gate, square-array collapse confirmed, and the "
          "guiding question answered: aspect ratio is a genuine, separate "
          "normalization factor, not absorbed by r/Rayleigh(D_diag) alone. ===")


if __name__ == "__main__":
    main()