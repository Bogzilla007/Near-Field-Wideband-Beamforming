"""
Stage 1 -- Core Array & Signal Setup

Provides:
    - ArrayGeometry: configurable ULA (ULA only for now; UPA optional later)
    - generate_ofdm_signal: wideband OFDM-style baseband test signal
    - generate_setup: convenience wrapper returning array + signal together

Design notes:
    - Everything here is vectorized with NumPy. No per-element Python loops.
    - Element spacing defaults to lambda/2 at the center frequency (standard,
      avoids grating lobes).
    - Frequencies are handled as absolute values (Hz), not normalized, so the
      near-field channel model (Stage 2) can use per-frequency wavelengths
      directly for the squint-aware / combined beamformers later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C = 299_792_458.0  # speed of light, m/s


@dataclass
class ArrayGeometry:
    """Uniform Linear Array (ULA) geometry.

    Attributes:
        N: number of elements.
        center_freq: carrier / center frequency in Hz (used to set spacing).
        spacing: element spacing in meters. Defaults to lambda/2 at center_freq.
        positions: 1D array of element positions along the array axis, meters,
            centered at 0 (array centroid at the origin).
        aperture: physical aperture size D = (N-1) * spacing, meters.
    """

    N: int
    center_freq: float
    spacing: float = None  # type: ignore[assignment]
    positions: np.ndarray = None  # type: ignore[assignment]
    aperture: float = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.N < 1:
            raise ValueError("N must be >= 1")
        if self.spacing is None:
            wavelength_c = C / self.center_freq
            self.spacing = wavelength_c / 2.0

        # Element indices 0..N-1, centered so the array centroid is at x=0.
        # positions[n] = d_n in the project doc's notation.
        idx = np.arange(self.N)
        self.positions = (idx - (self.N - 1) / 2.0) * self.spacing
        self.aperture = (self.N - 1) * self.spacing

    def rayleigh_distance(self, wavelength: float = None) -> float:
        """Analytical near-field/far-field boundary: R = 2*D^2/lambda.

        Args:
            wavelength: wavelength to evaluate at (meters). Defaults to the
                wavelength at center_freq.
        """
        if wavelength is None:
            wavelength = C / self.center_freq
        return 2.0 * self.aperture**2 / wavelength


def generate_ofdm_signal(
    bandwidth: float,
    center_freq: float,
    subcarrier_spacing: float = 120e3,
    n_symbols: int = 1,
    rng: np.random.Generator = None,
) -> dict:
    """Generate a wideband OFDM-style baseband test signal.

    This is not 3GPP-compliant -- it's a realistic-enough wideband OFDM
    structure (QPSK symbols on a set of active subcarriers spanning the
    requested bandwidth) with 5G-NR-like numerology as a reference point
    for subcarrier spacing.

    Args:
        bandwidth: total signal bandwidth in Hz (e.g. 100e6 - 1e9).
        center_freq: carrier center frequency in Hz.
        subcarrier_spacing: OFDM subcarrier spacing in Hz. 120 kHz is a
            5G-NR numerology value used for wide-bandwidth carriers.
        n_symbols: number of OFDM symbols to generate (kept small; this
            project needs a frequency-domain description of the signal
            more than a long time-domain waveform).
        rng: optional numpy random Generator for reproducibility.

    Returns:
        dict with:
            'freqs': (n_subcarriers,) absolute frequencies (Hz) of each
                active subcarrier, i.e. center_freq + subcarrier offsets.
            'symbols': (n_symbols, n_subcarriers) complex QPSK symbols.
            'subcarrier_spacing': echoed back for convenience.
            'bandwidth': echoed back for convenience.
            'center_freq': echoed back for convenience.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_subcarriers = int(np.floor(bandwidth / subcarrier_spacing))
    if n_subcarriers < 1:
        raise ValueError(
            "bandwidth too small relative to subcarrier_spacing "
            f"({bandwidth=}, {subcarrier_spacing=})"
        )

    # Subcarrier offsets centered around 0, then shifted to center_freq.
    # This vectorized construction avoids any per-subcarrier loop.
    k = np.arange(n_subcarriers) - (n_subcarriers - 1) / 2.0
    offsets = k * subcarrier_spacing
    freqs = center_freq + offsets

    # QPSK symbols, unit amplitude per subcarrier.
    bits = rng.integers(0, 4, size=(n_symbols, n_subcarriers))
    qpsk_lut = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    symbols = qpsk_lut[bits]

    return {
        "freqs": freqs,
        "symbols": symbols,
        "subcarrier_spacing": subcarrier_spacing,
        "bandwidth": bandwidth,
        "center_freq": center_freq,
    }


def generate_setup(
    N: int,
    bandwidth: float,
    center_freq: float,
    subcarrier_spacing: float = 120e3,
    n_symbols: int = 1,
    rng: np.random.Generator = None,
) -> dict:
    """Convenience wrapper: build array geometry + a wideband signal together.

    Returns:
        dict with keys 'array' (ArrayGeometry) and 'signal' (dict from
        generate_ofdm_signal).
    """
    array = ArrayGeometry(N=N, center_freq=center_freq)
    signal = generate_ofdm_signal(
        bandwidth=bandwidth,
        center_freq=center_freq,
        subcarrier_spacing=subcarrier_spacing,
        n_symbols=n_symbols,
        rng=rng,
    )
    return {"array": array, "signal": signal}