"""
Stage 14 -- Hybrid Analog + Digital Beamforming (fully-connected)

Fully-connected hybrid architecture: N_RF RF chains, each wired to ALL N
antenna elements through its OWN bank of N phase-only analog phase
shifters (total N*N_RF analog phase shifters). After the ADC, the
digital domain combines the N_RF chain outputs with a FULL complex
weight per chain, independently per frequency bin.

This is the standard "hybrid precoding" decomposition from mmWave MIMO
literature: the effective per-element, per-frequency beamforming vector
is

    a_eff[:, f] = V @ f_BB(f)

where V (N x N_RF) is the FIXED (frequency-flat, shared across the
whole band) analog phase-shifter matrix -- every column unit modulus,
phase-only, no amplitude control (a real hardware constraint) -- and
f_BB(f) is a FULLY FLEXIBLE complex digital weight vector, chosen
independently per frequency to best approximate the ideal per-element
near-field+squint corrector (the same per-element formula
combined_steering_vector/near_field_channel already use) via ordinary
least squares.

Design choice made explicit here (worth stating, since it drives what
this stage actually finds): analog columns are built from CURVATURE-
CORRECTED, frequency-flat phase at a set of "anchor" frequencies spread
across the signal band (one column per RF chain) -- i.e. each analog
beam is itself a near-field-aware, single-frequency phase-only steering
vector, just anchored at a different point in the band. This is a
simple, well-motivated, closed-form heuristic (least-squares digital
combining of these fixed candidate beams), not a full hybrid-precoding
optimization algorithm (OMP / alternating minimization) -- appropriate
for this stage's explicitly "kept loose, don't over-engineer" scope.

Key regression/convergence properties, both checked numerically in
validate_stage14.py (not just assumed):
    - N_RF=1 (single analog beam, anchored at center frequency) should
      exactly match the existing 'nearfield' mode's gain-loss --
      curvature-corrected but frequency-flat, same as that mode.
    - N_RF=N, with N distinct, generically-invertible analog columns,
      should recover the fully-digital 'combined' corrector's gain
      EXACTLY (0 dB loss) -- solving an invertible N x N linear system
      exactly, not approximately.
    - What happens IN BETWEEN is the genuinely interesting empirical
      question this stage tests: analog phase shifters can only ever
      contribute PHASE, never amplitude, so the near-field channel's
      amplitude taper (1/r_n) can only be reconstructed through
      constructive/destructive INTERFERENCE of multiple phase-only
      beams -- this may need meaningfully more RF chains than the
      squint/frequency-correction part of the problem does.
"""

from __future__ import annotations

import numpy as np

from arraymodel import ArrayGeometry, C
from channel import near_field_channel


def _select_anchor_freqs(freqs: np.ndarray, n_rf: int) -> np.ndarray:
    """Pick n_rf frequency anchors for the analog beams' phase design.

    If n_rf < len(freqs): n_rf evenly-spaced anchors sampled from the
    actual signal frequencies (guarantees at least one analog beam is
    well-matched near any point in the band).
    If n_rf >= len(freqs): cycle through ALL actual frequencies (every
    real signal frequency gets at least one exactly-anchored analog
    beam; extra RF chains beyond len(freqs) reuse anchors -- still
    useful, since each reused anchor becomes an independent column in
    the least-squares fit once combined with a different digital
    weight, but does not introduce a new frequency point).
    """
    freqs = np.atleast_1d(freqs)
    if n_rf >= len(freqs):
        idx = np.arange(n_rf) % len(freqs)
    elif n_rf == 1:
        # Special-cased so the N_RF=1 regression check is exact: a single
        # analog beam anchored at the CENTER frequency should reproduce
        # the existing 'nearfield' mode exactly (curvature-corrected,
        # frequency-flat) -- np.linspace(0, len-1, 1) would otherwise pick
        # index 0 (the band's lowest frequency), not the center, breaking
        # this regression property for no good reason.
        idx = np.array([len(freqs) // 2])
    else:
        idx = np.linspace(0, len(freqs) - 1, n_rf).round().astype(int)
    return freqs[idx]


def _build_analog_matrix(array: ArrayGeometry, r: float, theta: float, anchor_freqs: np.ndarray) -> np.ndarray:
    """Build the (N, N_RF) analog phase-shifter matrix: each column is a
    unit-modulus, curvature-corrected, frequency-flat steering vector
    anchored at one frequency. Phase-only -- no amplitude control,
    matching real analog phase-shifter hardware.
    """
    d_n = array.positions[:, None]  # (N, 1)
    r_n = np.sqrt(r**2 + d_n**2 - 2 * r * d_n * np.sin(theta))  # (N, 1), exact curvature
    f_anchors = np.atleast_1d(anchor_freqs)[None, :]  # (1, N_RF)
    phase = 2 * np.pi * f_anchors / C * r_n  # (N, N_RF)
    return np.exp(-1j * phase)  # unit modulus


def hybrid_effective_steering(
    array: ArrayGeometry,
    r: float,
    theta: float,
    freqs: np.ndarray,
    n_rf: int,
) -> np.ndarray:
    """Build the effective (N, F) steering vector achievable by a
    fully-connected hybrid architecture with n_rf RF chains, targeting
    the known (r, theta) location.

    Args:
        array: ArrayGeometry.
        r, theta: target location (same convention as every other mode).
        freqs: (F,) frequencies the signal occupies, Hz.
        n_rf: number of RF chains (equivalently, number of analog beams).

    Returns:
        (N, F) complex effective steering vector, energy-normalized to
        sum_n|a_eff[n,f]|^2 = N per frequency column -- same convention
        as every other mode in beamformers.py, for direct comparability.
    """
    freqs = np.atleast_1d(freqs)
    anchor_freqs = _select_anchor_freqs(freqs, n_rf)
    V = _build_analog_matrix(array, r, theta, anchor_freqs)  # (N, N_RF)

    # Ideal per-element, per-frequency target: the same raw formula
    # near_field_channel/combined_steering_vector already use (before
    # their own final normalization) -- (1/r_n)*exp(-j*2*pi*f/c*r_n).
    a_ideal = near_field_channel(array, r, theta, freqs)  # (N, F)

    # Least-squares digital weight per frequency: f_BB(f) = argmin_c
    # ||a_ideal[:,f] - V c||^2, solved via the standard least-squares
    # pseudo-inverse solution (exact solve, not an iterative approximation,
    # when V is square and invertible -- e.g. at n_rf=N with generically
    # distinct anchor phases).
    f_BB, *_ = np.linalg.lstsq(V, a_ideal, rcond=None)  # (N_RF, F)

    a_eff = V @ f_BB  # (N, F)

    # Same per-frequency energy normalization convention as every other
    # mode (combined_steering_vector, nearfield_steering_vector).
    N = array.N
    energy_per_freq = np.sum(np.abs(a_eff) ** 2, axis=0, keepdims=True)
    a_eff = a_eff * np.sqrt(N / energy_per_freq)
    return a_eff


def cost_estimate_hybrid(N: int, n_rf: int, n_freq_bins: int) -> dict:
    """Rough hardware/compute cost for the fully-connected hybrid
    architecture, mirroring correction.py's cost_estimate for the
    fully-digital combined corrector.

    Fully-connected: each of the n_rf RF chains needs its own bank of N
    analog phase shifters (N*n_rf total, fixed hardware, frequency-flat
    -- built once, not per beam-update). Digital combining needs one
    complex multiply per RF chain per frequency bin per beam-update
    (n_rf * n_freq_bins).
    """
    analog_phase_shifters = N * n_rf  # fixed hardware, not per-update
    digital_multiplies_per_update = n_rf * n_freq_bins
    combined_digital_multiplies_per_update = N * n_freq_bins  # from correction.py, for comparison
    return {
        "N": N,
        "n_rf": n_rf,
        "n_freq_bins": n_freq_bins,
        "analog_phase_shifters": analog_phase_shifters,
        "digital_multiplies_per_update": digital_multiplies_per_update,
        "combined_digital_multiplies_per_update": combined_digital_multiplies_per_update,
        "digital_cost_ratio_vs_combined": digital_multiplies_per_update / combined_digital_multiplies_per_update,
    }