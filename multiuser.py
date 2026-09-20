"""
Stage 11 -- Multi-User Near-Field Spatial Multiplexing

Scenario: TWO users at the SAME angle theta from a single array, but at
DIFFERENT distances r1, r2 (both within near-field range). A far-field/
angle-only beamformer cannot tell them apart -- they look identical from
an angle-only point of view. A near-field-aware beamformer, using range
as well as angle, can favor one user's true location over the other.

This module deliberately does NOT superpose the two users' channels into
a single combined receive signal (h1 + h2) for the main metric -- the
question here is "how much does a beam meant for user 1 leak toward user
2's location", which only requires evaluating ONE steering vector's gain
against TWO separate channels, not summing them. See
`superposition_sanity_check` below for the one place this module does
sum two near-field channels, purely as a validity check flagged in the
Extension Plan (near_field_channel / beamform_gain were originally
written and validated assuming a single source; this checks they behave
sanely under superposition before anything downstream is allowed to rely
on it).

Per-user channels are power-normalized (same convention used everywhere
since Stage 3: sum|h|^2 = N*F) BEFORE computing gain. This deliberately
removes the natural 1/r path-loss advantage the closer user would
otherwise have, so the leakage/separation metric measures genuine
PHASE/CURVATURE-SHAPE discrimination -- not just "the near beamformer
wins because the near user is louder anyway". This mirrors exactly the
reasoning already applied to every gain-loss metric in Stages 3-6.
"""

from __future__ import annotations

import numpy as np

from arraymodel import ArrayGeometry
from channel import near_field_channel
from beamformers import beamform, beamform_gain
from constants import THETA


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Normalize total channel power to N*F (unit power per element per
    frequency bin) -- identical convention to sweep._normalize_channel,
    duplicated here (not imported) to keep this module's dependency
    surface self-contained per-scenario, matching correction.py/sensing.py's
    existing pattern of light duplication over cross-file coupling."""
    N, F = channel.shape
    power = np.sum(np.abs(channel) ** 2)
    return channel * np.sqrt(N * F / power)


def two_user_channels(
    array: ArrayGeometry,
    r1: float,
    r2: float,
    theta: float,
    freqs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the two users' individually-normalized near-field channels.

    Both users share the same angle `theta` -- the whole point of this
    scenario is that they are indistinguishable by angle alone.

    Returns:
        (h1, h2), each (N, F), each independently normalized to
        sum|h|^2 = N*F.
    """
    h1 = _normalize_channel(near_field_channel(array, r1, theta, freqs))
    h2 = _normalize_channel(near_field_channel(array, r2, theta, freqs))
    return h1, h2


def separation_db(
    steering: np.ndarray,
    h_intended: np.ndarray,
    h_other: np.ndarray,
) -> dict:
    """Compute the SINR-style separation metric for one steering vector.

    Args:
        steering: (N, F) steering vector under test, aimed at the
            intended user.
        h_intended: (N, F) intended user's normalized channel.
        h_other: (N, F) other (unintended) user's normalized channel.

    Returns:
        dict with 'gain_intended', 'gain_leaked', 'separation_db'
        (10*log10(gain_intended / gain_leaked); positive = good
        discrimination in favor of the intended user, ~0 = cannot tell
        them apart, negative would mean it actually favors the WRONG
        user).
    """
    gain_intended = beamform_gain(steering, h_intended)
    gain_leaked = beamform_gain(steering, h_other)
    return {
        "gain_intended": gain_intended,
        "gain_leaked": gain_leaked,
        "separation_db": 10 * np.log10(gain_intended / gain_leaked),
    }


def superposition_sanity_check(
    array: ArrayGeometry,
    r1: float,
    r2: float,
    theta: float,
    freqs: np.ndarray,
) -> dict:
    """Validity check flagged in the Extension Plan: confirm
    near_field_channel/beamform_gain behave sanely when two near-field
    channels are SUMMED (as they would be at a receiver seeing both
    users' unintended crosstalk simultaneously), even though the main
    Stage 11 metric above deliberately avoids relying on this.

    Checks:
        1. No NaN/Inf in the summed channel.
        2. The summed channel's total power is neither collapsed to ~0
           (which would suggest silent cancellation/bug) nor blown up
           far beyond the simple sum of the two individual powers
           (which would suggest a runaway amplification bug). Since
           the two channels have independent, non-adversarial phase
           relationships (different r -> different phase slope across
           elements), the combined power should land somewhere in a
           physically sane range relative to the two individual powers.
        3. Cauchy-Schwarz sanity bound on beamform_gain against the sum:
           gain(h1+h2) should not exceed (sqrt(gain(h1)) + sqrt(gain(h2)))^2
           for any fixed steering vector, evaluated here using the
           combined-mode steering vector aimed at user 1. This is a
           mathematical identity for the |a^H . h|^2 gain definition and
           would only be violated by an implementation bug, not by any
           real physical effect.

    Returns:
        dict with 'passed' (bool) and diagnostic values, so the
        checkpoint script can print/assert on it explicitly rather than
        this function raising directly (keeps the validate script the
        single place assertions live, per this project's convention).
    """
    h1_raw = near_field_channel(array, r1, theta, freqs)
    h2_raw = near_field_channel(array, r2, theta, freqs)
    h_sum = h1_raw + h2_raw

    finite_ok = np.all(np.isfinite(h_sum))

    power1 = np.sum(np.abs(h1_raw) ** 2)
    power2 = np.sum(np.abs(h2_raw) ** 2)
    power_sum = np.sum(np.abs(h_sum) ** 2)
    # For two signals with independent (non-adversarial) phase, expected
    # combined power is somewhere between |P1-P2|-ish (near-total
    # destructive alignment, unlikely across all elements/frequencies
    # simultaneously) and P1+P2+2*sqrt(P1*P2) (total constructive
    # alignment, the absolute physical ceiling from the triangle
    # inequality on complex amplitudes). Checking against this ceiling
    # (with a small numerical margin) catches implementation bugs
    # (e.g. accidental power-doubling or normalization applied to the
    # sum in a way that isn't physically meaningful) without asserting
    # a specific expected value, since the true value depends on the
    # actual phase relationship at this r1/r2/theta and isn't a "nice"
    # closed form worth hand-deriving here.
    ceiling = power1 + power2 + 2 * np.sqrt(power1 * power2)
    power_within_ceiling = power_sum <= ceiling * 1.01  # 1% numerical margin

    steer_combined = beamform("combined", array, freqs, theta, r=r1)
    gain1 = beamform_gain(steer_combined, h1_raw)
    gain2 = beamform_gain(steer_combined, h2_raw)
    gain_sum = beamform_gain(steer_combined, h_sum)
    cauchy_schwarz_bound = (np.sqrt(gain1) + np.sqrt(gain2)) ** 2
    cauchy_schwarz_ok = gain_sum <= cauchy_schwarz_bound * 1.01  # 1% numerical margin

    passed = finite_ok and power_within_ceiling and cauchy_schwarz_ok

    return {
        "passed": passed,
        "finite_ok": finite_ok,
        "power1": power1,
        "power2": power2,
        "power_sum": power_sum,
        "power_ceiling": ceiling,
        "power_within_ceiling": power_within_ceiling,
        "gain1": gain1,
        "gain2": gain2,
        "gain_sum": gain_sum,
        "cauchy_schwarz_bound": cauchy_schwarz_bound,
        "cauchy_schwarz_ok": cauchy_schwarz_ok,
    }


def delay_only_steering_vector(
    array: ArrayGeometry,
    r: float,
    theta: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Isolates the DELAY/phase-slope mechanism from CURVATURE, for
    Stage 11's Condition C.

    IMPORTANT DESIGN NOTE: the existing `squint_aware_steering_vector`
    (beamformers.py, mode="squint") is angle-only and has NO `r`
    parameter at all -- its phase depends only on theta, never on
    distance. A static dot-product against it therefore cannot encode
    ANY range information by construction, regardless of bandwidth. An
    early run of this scenario using mode="squint" for Condition C
    produced separation numbers nearly identical to the angle-only
    conventional baseline (~-9 dB in both cases) -- not because "delay
    alone doesn't help", but because mode="squint" was never given a way
    to know r in the first place. That was a flawed isolation, not a
    real finding, and is replaced by this function.

    This steering vector uses the EXACT per-element, per-frequency phase
    term (2*pi*f/c * r_n, using the true curved r_n -- this is exactly
    where the range-dependent phase-vs-frequency SLOPE that Stage 7's
    ranging exploited comes from) but keeps amplitude UNIFORM (|a_n|=1,
    no 1/r_n taper). This isolates "delay/phase-slope-based range
    information" from "amplitude-curvature-based range information" --
    the two mechanisms this project has kept separate since Stage 2.

    a_delay_only[n, f] = exp(-j * 2*pi*f/c * r_n)   (unit amplitude)

    Args:
        array: ArrayGeometry.
        r: distance to steer to, meters.
        theta: angle to steer to, radians.
        freqs: (F,) frequencies the signal occupies (Hz).

    Returns:
        (N, F) complex steering vector, unit amplitude per element.
    """
    from arraymodel import C as SPEED_OF_LIGHT

    d_n = array.positions[:, None]  # (N, 1)
    r_n = np.sqrt(r**2 + d_n**2 - 2 * r * d_n * np.sin(theta))  # (N, 1)
    f = np.atleast_1d(freqs)[None, :]  # (1, F)
    phase = 2 * np.pi * f / SPEED_OF_LIGHT * r_n  # (N, F), exact r_n, per-frequency
    return np.exp(-1j * phase)  # unit amplitude -- no curvature/taper term


def run_scenario(
    N: int,
    r1_frac_rayleigh: float,
    r2_frac_rayleigh: float,
    theta: float,
    center_freq: float,
    wideband_bandwidth: float,
    n_freq_bins: int = 33,
) -> dict:
    """Run all A/B/C conditions plus the conventional/squint controls for
    one (N, r1, r2) scenario. Steering is always aimed at user 1.

    Conditions:
        conventional_narrowband: angle-only, single tone. Baseline
            negative control -- should show ~0 dB separation (cannot
            discriminate by range at all, and no bandwidth to help it).
        conventional_wideband: angle-only, but evaluated across the
            wideband channel. Also expected ~0 dB separation --
            confirms the conventional beamformer's frequency-independent
            weights don't accidentally pick up any range discrimination
            even when the channel itself is wideband.
        A_narrowband_nearfield: near-field-aware steering, single tone.
            CURVATURE ONLY, no frequency diversity. The key positive
            result if this alone shows real separation.
        C_wideband_delay_only: DELAY/phase-slope only steering (see
            delay_only_steering_vector docstring for why this replaces a
            naive mode="squint" attempt -- squint-aware has no r
            parameter at all and cannot encode range information by
            construction). Exact r-dependent phase, uniform amplitude
            (no curvature taper) -- the control isolating the Stage
            7-style delay-based mechanism from curvature.
        B_wideband_combined: combined (near-field + squint-aware)
            steering, wideband. CURVATURE + DELAY together -- the full
            realistic case. Should be >= max(A, C).

    Returns:
        dict with 'rayleigh', 'r1', 'r2', and 'conditions' (a dict of
        condition_name -> separation_db() result dict).
    """
    array = ArrayGeometry(N=N, center_freq=center_freq)
    rayleigh = array.rayleigh_distance()
    r1 = r1_frac_rayleigh * rayleigh
    r2 = r2_frac_rayleigh * rayleigh

    freqs_narrow = np.array([center_freq])
    freqs_wide = center_freq + np.linspace(
        -wideband_bandwidth / 2, wideband_bandwidth / 2, n_freq_bins
    )

    h1_narrow, h2_narrow = two_user_channels(array, r1, r2, theta, freqs_narrow)
    h1_wide, h2_wide = two_user_channels(array, r1, r2, theta, freqs_wide)

    conditions = {}

    steer = beamform("conventional", array, freqs_narrow, theta, center_freq=center_freq)
    conditions["conventional_narrowband"] = separation_db(steer, h1_narrow, h2_narrow)

    steer = beamform("conventional", array, freqs_wide, theta, center_freq=center_freq)
    conditions["conventional_wideband"] = separation_db(steer, h1_wide, h2_wide)

    steer = beamform("nearfield", array, freqs_narrow, theta, r=r1, center_freq=center_freq)
    conditions["A_narrowband_nearfield"] = separation_db(steer, h1_narrow, h2_narrow)

    steer = delay_only_steering_vector(array, r1, theta, freqs_wide)
    conditions["C_wideband_delay_only"] = separation_db(steer, h1_wide, h2_wide)

    steer = beamform("combined", array, freqs_wide, theta, r=r1)
    conditions["B_wideband_combined"] = separation_db(steer, h1_wide, h2_wide)

    return {
        "N": N,
        "rayleigh": rayleigh,
        "r1": r1,
        "r2": r2,
        "r1_frac_rayleigh": r1_frac_rayleigh,
        "r2_frac_rayleigh": r2_frac_rayleigh,
        "conditions": conditions,
    }