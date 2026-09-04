"""
Stage 3/4 -- Beamformer Modes (Conventional, Near-Field-Aware, Squint-Aware, Combined)

Stage 3 provides:
    - conventional_steering_vector: single frequency-independent phase
      steering vector (the "broken at scale" baseline), applied uniformly
      across the whole signal bandwidth -- deliberately ignores frequency
      dependence, which is exactly what makes it vulnerable to beam squint
      later (Stage 4/5).
    - beamform_gain: computes Gain = |steering_vector^H . channel|^2 per
      the project's metric definition (Section 2.6), evaluated per
      frequency and summed/averaged across the signal band.

Stage 4 adds three more modes, all sharing the same (N, F) interface so
Stage 5's sweep engine can call any of the four interchangeably via
`beamform(mode=..., ...)`:
    - nearfield_steering_vector: corrects near-field curvature only.
    - squint_aware_steering_vector: corrects beam squint only (TTD).
    - combined_steering_vector: corrects both simultaneously -- this is
      the "ideal" reference beamformer used throughout the project's
      gain-loss metric (Section 2.5b).

Gain convention (pinned down explicitly, per project doc Section 2.6):
    Gain = |a^H . h|^2, using the RAW (unnormalized, unit-per-element-
    amplitude) steering vector dotted with the channel vector. With a
    matched, unit-amplitude-per-element steering vector against a
    far-field channel of the same convention, coherent combining across
    N elements gives |a^H . h|^2 = |N|^2 = N^2 -- this is the classical
    "power pattern peak" scaling, not the N-scaling of an SNR-normalized
    array gain (which would come from a 1/N-normalized steering vector
    instead). This project uses the UNNORMALIZED convention, validated
    against N^2 in validate_stage3.py.
"""

from __future__ import annotations

import numpy as np

from arraymodel import C, ArrayGeometry


def conventional_steering_vector(
    array: ArrayGeometry,
    theta: float,
    freqs: np.ndarray,
    center_freq: float,
) -> np.ndarray:
    """Conventional phase-shifter steering vector.

    A single phase shift per element, computed at the CENTER frequency
    only, then applied unchanged across every frequency in `freqs`. This
    is what makes it "conventional" -- it has no frequency dependence,
    which is correct at the carrier but increasingly wrong for
    off-carrier frequencies as bandwidth grows (beam squint, Stage 5).

    a_conventional[n, f] = exp(j * 2*pi*center_freq/c * d_n * sin(theta))
                            for every f (no f-dependence)

    Args:
        array: ArrayGeometry.
        theta: steering angle, radians.
        freqs: (F,) frequencies the signal occupies (Hz). Only used to
            determine output shape -- the phase itself does not depend on
            these, by construction.
        center_freq: carrier frequency (Hz) the phase shift is computed at.

    Returns:
        (N, F) complex steering vector, constant across the F axis.
    """
    d_n = array.positions[:, None]  # (N, 1)
    phase = 2 * np.pi * center_freq / C * d_n * np.sin(theta)  # (N, 1), no f dependence
    a_single_freq = np.exp(1j * phase)  # (N, 1)
    F = np.atleast_1d(freqs).shape[0]
    return np.repeat(a_single_freq, F, axis=1)  # (N, F), broadcast across frequency


def beamform_gain(steering: np.ndarray, channel: np.ndarray) -> float:
    """Compute beamforming gain per the project's Section 2.6 metric.

    Gain = |steering^H . channel|^2, evaluated per-frequency and averaged
    across the frequency axis (so multi-frequency wideband channels/
    steering vectors reduce to a single scalar gain figure, consistent
    with how this will be used across narrowband and wideband cases alike
    in later stages).

    Args:
        steering: (N, F) steering vector used by the beamformer under test.
        channel: (N, F) true channel response (always the exact near-field,
            per-frequency model -- see project doc Section 2.6).

    Returns:
        Scalar gain (average over frequency bins of |a^H h|^2 per bin).
    """
    # Per-frequency inner product: sum over elements (axis 0), for each freq.
    inner = np.sum(np.conj(steering) * channel, axis=0)  # (F,)
    gain_per_freq = np.abs(inner) ** 2
    return float(np.mean(gain_per_freq))


# ---------------------------------------------------------------------------
# Stage 4 -- Advanced Beamforming (Near-Field-Aware + Squint-Aware + Combined)
# ---------------------------------------------------------------------------

def nearfield_steering_vector(
    array: ArrayGeometry,
    r: float,
    theta: float,
    freqs: np.ndarray,
    center_freq: float,
) -> np.ndarray:
    """Near-field-aware steering vector.

    Corrects for near-field curvature (exact per-element distance r_n,
    both phase AND amplitude) but still uses only the CENTER frequency
    for the phase term, repeated across all frequency bins -- i.e. it
    corrects near-field error but does NOT correct beam squint. This is
    deliberate: per the project doc's isolation principle (Section 2.5),
    each corrector should fix exactly one mechanism so its correctness
    can be validated in isolation before combining.

    a_nearfield_aware[n, f] = (1/r_n) * exp(-j * 2*pi*center_freq/c * r_n)
                               for every f (no f-dependence -> still
                               squint-vulnerable)

    Sign/amplitude convention matches near_field_channel in channel.py, so
    a beamformer using this steering vector against a matched near-field
    channel (at the center frequency) recovers the same N^2-style peak
    gain as the conventional beamformer does for far-field (Stage 3).

    Args:
        array: ArrayGeometry.
        r: distance to steer to, meters.
        theta: angle to steer to, radians.
        freqs: (F,) frequencies the signal occupies (Hz). Only used for
            output shape -- the phase itself does not depend on these.
        center_freq: carrier frequency (Hz) the phase/distance term is
            computed at.

    Returns:
        (N, F) complex steering vector, constant across the F axis.
    """
    d_n = array.positions[:, None]  # (N, 1)
    r_n = np.sqrt(r**2 + d_n**2 - 2 * r * d_n * np.sin(theta))  # (N, 1)
    amplitude = 1.0 / r_n
    phase = 2 * np.pi * center_freq / C * r_n  # (N, 1), center-freq only
    a_single_freq = amplitude * np.exp(-1j * phase)  # (N, 1)

    # Normalize to unit RMS amplitude per element (sum|a_n|^2 = N), matching
    # the constant-modulus convention conventional/squint-aware steering
    # vectors have implicitly (|a_n| = 1 there). Without this, the doc's raw
    # 1/r_n amplitude taper gives this steering vector a DIFFERENT total
    # energy than the other modes, which unfairly inflates or deflates the
    # unnormalized gain metric |a^H h|^2 -- this is not a beamforming
    # correctness difference, it's a scale artifact. Caught by Stage 4's
    # checkpoint: conventional briefly scored HIGHER gain than the combined
    # corrector, which is structurally impossible for a matched-filter
    # reference and was the signal something was wrong here.
    N = array.N
    energy = np.sum(np.abs(a_single_freq) ** 2)
    a_single_freq = a_single_freq * np.sqrt(N / energy)

    F = np.atleast_1d(freqs).shape[0]
    return np.repeat(a_single_freq, F, axis=1)  # (N, F)


def squint_aware_steering_vector(
    array: ArrayGeometry,
    theta: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Squint-aware (true-time-delay / TTD) steering vector.

    Corrects beam squint by applying a genuine per-element, per-FREQUENCY
    phase term (equivalent to a true time delay) rather than a single
    frequency-independent phase shift -- so every frequency component in
    the wideband signal steers to the same angle. Does NOT correct for
    near-field curvature (angle-only steering, no distance term) -- this
    is deliberate isolation, matching the doc's Section 2.5 principle.

    This is mathematically identical in form to far_field_channel (both
    vary phase linearly with d_n * sin(theta) at each f), which is exactly
    the doc's point: TTD beamforming IS the frequency-correct version of
    the far-field/angle steering model. Implemented as its own function
    (rather than a bare alias) so Stage 5's mode dispatch and this file's
    docstrings stay self-explanatory.

    a_squint_aware[n, f] = exp(j * 2*pi*f/c * d_n * sin(theta))

    Args:
        array: ArrayGeometry.
        theta: steering angle, radians.
        freqs: (F,) frequencies the signal occupies (Hz).

    Returns:
        (N, F) complex steering vector, genuinely varying across F.
    """
    d_n = array.positions[:, None]        # (N, 1)
    f = np.atleast_1d(freqs)[None, :]     # (1, F)
    phase = 2 * np.pi * f / C * d_n * np.sin(theta)
    return np.exp(1j * phase)


def combined_steering_vector(
    array: ArrayGeometry,
    r: float,
    theta: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Combined near-field + squint-aware corrector (project doc Section 2.5b).

    The "ideal" reference beamformer for every gain-loss metric in this
    project: uses the exact per-element spherical distance r_n AND a true
    per-element, per-frequency phase term simultaneously. No remaining
    model mismatch against the true near-field, wideband channel.

    a_combined[n, f] = (1/r_n) * exp(-j * 2*pi*f/c * r_n)

    This has the same functional form as near_field_channel in channel.py
    (per-frequency near-field model) -- reused directly rather than
    reimplemented, since the combined corrector IS the exact channel
    model evaluated at the target (r, theta). Wrapped as its own named
    function here (rather than just importing near_field_channel
    directly in Stage 5) so the beamformer-mode interface stays uniform
    and self-documenting.

    Args:
        array: ArrayGeometry.
        r: distance to steer to, meters.
        theta: angle to steer to, radians.
        freqs: (F,) frequencies the signal occupies (Hz).

    Returns:
        (N, F) complex steering vector.
    """
    from channel import near_field_channel  # local import avoids a cycle at module load
    a = near_field_channel(array, r, theta, freqs)  # (N, F)

    # Same normalization rationale as nearfield_steering_vector: rescale
    # EACH FREQUENCY COLUMN independently to sum|a_n|^2 = N, so this mode's
    # energy matches the other three modes' convention at every frequency
    # bin (not just on average across the band).
    N = array.N
    energy_per_freq = np.sum(np.abs(a) ** 2, axis=0, keepdims=True)  # (1, F)
    a = a * np.sqrt(N / energy_per_freq)
    return a


def beamform(
    mode: str,
    array: ArrayGeometry,
    freqs: np.ndarray,
    theta: float,
    r: float = None,
    center_freq: float = None,
) -> np.ndarray:
    """Uniform dispatch across all four beamformer modes (Stage 4 deliverable).

    Keeps identical input/output signature across modes so Stage 5's sweep
    engine can call any of them interchangeably.

    Args:
        mode: one of "conventional", "nearfield", "squint", "combined".
        array: ArrayGeometry.
        freqs: (F,) frequencies the signal occupies (Hz).
        theta: target steering angle, radians.
        r: target steering distance, meters. Required for "nearfield" and
            "combined" modes.
        center_freq: carrier frequency, Hz. Required for "conventional"
            and "nearfield" modes.

    Returns:
        (N, F) complex steering vector.
    """
    if mode == "conventional":
        if center_freq is None:
            raise ValueError("conventional mode requires center_freq")
        return conventional_steering_vector(array, theta, freqs, center_freq)
    elif mode == "nearfield":
        if r is None or center_freq is None:
            raise ValueError("nearfield mode requires r and center_freq")
        return nearfield_steering_vector(array, r, theta, freqs, center_freq)
    elif mode == "squint":
        return squint_aware_steering_vector(array, theta, freqs)
    elif mode == "combined":
        if r is None:
            raise ValueError("combined mode requires r")
        return combined_steering_vector(array, r, theta, freqs)
    else:
        raise ValueError(f"Unknown beamformer mode: {mode!r}")