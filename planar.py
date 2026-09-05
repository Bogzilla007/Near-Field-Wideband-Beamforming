"""
Stage 12 -- 2D Uniform Planar Array (UPA)

Generalizes the 1D ULA model (arraymodel.py/channel.py/beamformers.py)
to a 2D grid array. Kept as a SEPARATE module (rather than modifying the
shared 1D core files every earlier stage depends on) -- following this
project's established pattern of adding one file per stage
(hardware.py, noise.py, multiuser.py) rather than risking the existing,
already-validated 1D pipeline.

Guiding question (per the Extension Plan v2): not just "does the 1D
model generalize", but "what's the correct normalization variable in
2D?" In 1D, r/Rayleigh(N) was the single scalar that made Stage 2's
divergence curves collapse across every array size. In 2D there are TWO
array dimensions (N_x, N_y) and TWO angles (azimuth theta, elevation
phi) -- it is not assumed here that a single scalar still does the job.
validate_stage12.py tests this directly rather than assuming it.

Geometry convention:
    Array lies in the x-y plane, boresight along +z. Element positions
    are (x_n, y_n, 0), on a regular grid, lambda/2 spacing in both
    dimensions (matching the 1D ULA's spacing convention).

    User direction is parameterized by (r, theta, phi):
        theta = azimuth (radians, 0 = broadside in x)
        phi   = elevation (radians, 0 = broadside in y)
        target position = r * (sin(theta)*cos(phi), sin(phi), cos(theta)*cos(phi))

    This choice is deliberately constructed so that setting N_y=1
    (single row along x) and phi=0 makes the 2D model reduce EXACTLY
    (to machine precision) to the existing 1D ULA formulas -- this is
    Stage 12's hard regression gate, checked in validate_stage12.py
    before anything else here is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arraymodel import C


@dataclass
class PlanarArrayGeometry:
    """2D Uniform Planar Array (UPA) geometry, on an N_x x N_y grid.

    Attributes:
        N_x, N_y: elements along each grid dimension. Total N = N_x*N_y.
        center_freq: carrier frequency in Hz (sets spacing).
        spacing: element spacing in meters, both dimensions. Defaults to
            lambda/2 at center_freq (same convention as the 1D ULA).
        positions: (N, 2) array of (x_n, y_n) element positions, meters,
            centered so the array centroid is at the origin.
        aperture_x, aperture_y: physical aperture in each dimension.
        aperture_diag: sqrt(aperture_x^2 + aperture_y^2) -- the diagonal
            aperture, used for the 2D Rayleigh distance (see
            rayleigh_distance() docstring for why this choice, and its
            regression-tested reduction to the 1D case).
    """

    N_x: int
    N_y: int
    center_freq: float
    spacing: float = None  # type: ignore[assignment]
    positions: np.ndarray = None  # type: ignore[assignment]
    aperture_x: float = None  # type: ignore[assignment]
    aperture_y: float = None  # type: ignore[assignment]
    aperture_diag: float = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.N_x < 1 or self.N_y < 1:
            raise ValueError("N_x and N_y must both be >= 1")
        if self.spacing is None:
            wavelength_c = C / self.center_freq
            self.spacing = wavelength_c / 2.0

        idx_x = np.arange(self.N_x) - (self.N_x - 1) / 2.0
        idx_y = np.arange(self.N_y) - (self.N_y - 1) / 2.0
        # Grid ordering: y varies fastest within each x row is arbitrary --
        # fixed here as x-major (outer), y-minor (inner) via meshgrid +
        # ravel, consistent throughout this module.
        X, Y = np.meshgrid(idx_x * self.spacing, idx_y * self.spacing, indexing="ij")
        self.positions = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N, 2)

        self.aperture_x = (self.N_x - 1) * self.spacing
        self.aperture_y = (self.N_y - 1) * self.spacing
        self.aperture_diag = np.sqrt(self.aperture_x**2 + self.aperture_y**2)

    @property
    def N(self) -> int:
        return self.N_x * self.N_y

    def rayleigh_distance(self, wavelength: float = None) -> float:
        """2D Rayleigh distance: R = 2*D_diag^2/lambda, using the
        DIAGONAL aperture D_diag = sqrt(aperture_x^2 + aperture_y^2).

        This is the standard textbook extension of the 1D formula to a
        2D aperture. By construction it reduces EXACTLY to the 1D
        formula when N_y=1 (aperture_y=0, so D_diag=aperture_x) --
        checked directly in validate_stage12.py's regression gate.
        Whether this diagonal-based scalar is ALSO the correct
        normalization variable for off-axis (theta, phi != 0) or
        non-square (N_x != N_y) cases is an open empirical question,
        tested (not assumed) in validate_stage12.py's Part 3.
        """
        if wavelength is None:
            wavelength = C / self.center_freq
        return 2.0 * self.aperture_diag**2 / wavelength


def _target_position(r: float, theta: float, phi: float) -> np.ndarray:
    """(x, y, z) target position from (r, theta, phi), per this module's
    geometry convention (see module docstring)."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(phi)
    z = r * np.cos(theta) * np.cos(phi)
    return np.array([x, y, z])


def near_field_channel_planar(
    array: PlanarArrayGeometry,
    r: float,
    theta: float,
    phi: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Exact near-field spherical channel model, 2D UPA version.

    r_n = || target_position - (x_n, y_n, 0) ||
    a_nearfield[n, f] = (1/r_n) * exp(-j * 2*pi*f/c * r_n)

    Reduces EXACTLY to channel.near_field_channel when N_y=1, phi=0 --
    checked in validate_stage12.py.

    Args:
        array: PlanarArrayGeometry.
        r, theta, phi: user position (see module docstring).
        freqs: (F,) frequencies, Hz.

    Returns:
        (N, F) complex channel matrix, N = array.N.
    """
    target = _target_position(r, theta, phi)  # (3,)
    x_n = array.positions[:, 0]
    y_n = array.positions[:, 1]
    r_n = np.sqrt(
        (target[0] - x_n) ** 2 + (target[1] - y_n) ** 2 + target[2] ** 2
    )[:, None]  # (N, 1)
    f = np.atleast_1d(freqs)[None, :]  # (1, F)
    amplitude = 1.0 / r_n
    phase = 2 * np.pi * f / C * r_n
    return amplitude * np.exp(-1j * phase)


def far_field_channel_planar(
    array: PlanarArrayGeometry,
    theta: float,
    phi: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Far-field planar-wavefront channel model, 2D UPA version.

    Standard planar-array far-field phase: linear in the dot product of
    element position with the unit direction vector.

    a_farfield[n, f] = exp(j * 2*pi*f/c * (x_n*sin(theta)*cos(phi) + y_n*sin(phi)))

    Reduces EXACTLY to channel.far_field_channel when N_y=1, phi=0 --
    checked in validate_stage12.py.
    """
    x_n = array.positions[:, 0][:, None]  # (N, 1)
    y_n = array.positions[:, 1][:, None]  # (N, 1)
    f = np.atleast_1d(freqs)[None, :]     # (1, F)
    phase = 2 * np.pi * f / C * (x_n * np.sin(theta) * np.cos(phi) + y_n * np.sin(phi))
    return np.exp(1j * phase)


def _normalize_energy(a: np.ndarray) -> np.ndarray:
    """Rescale each frequency column to sum_n|a_n|^2 = N, matching the
    1D beamformers.py convention exactly (needed for fair gain
    comparisons across modes, same rationale as Stage 4)."""
    N = a.shape[0]
    energy_per_freq = np.sum(np.abs(a) ** 2, axis=0, keepdims=True)
    return a * np.sqrt(N / energy_per_freq)


def conventional_steering_vector_planar(
    array: PlanarArrayGeometry,
    theta: float,
    phi: float,
    freqs: np.ndarray,
    center_freq: float,
) -> np.ndarray:
    """Conventional (angle-only, center-frequency-only phase) steering
    vector, 2D UPA version. Mirrors beamformers.conventional_steering_vector."""
    x_n = array.positions[:, 0][:, None]
    y_n = array.positions[:, 1][:, None]
    phase = 2 * np.pi * center_freq / C * (x_n * np.sin(theta) * np.cos(phi) + y_n * np.sin(phi))
    a_single_freq = np.exp(1j * phase)  # (N, 1), unit modulus -> already sum|a_n|^2=N
    F = np.atleast_1d(freqs).shape[0]
    return np.repeat(a_single_freq, F, axis=1)


def nearfield_steering_vector_planar(
    array: PlanarArrayGeometry,
    r: float,
    theta: float,
    phi: float,
    freqs: np.ndarray,
    center_freq: float,
) -> np.ndarray:
    """Near-field-aware (curvature-corrected, center-frequency-only phase)
    steering vector, 2D UPA version. Mirrors
    beamformers.nearfield_steering_vector."""
    target = _target_position(r, theta, phi)
    x_n = array.positions[:, 0]
    y_n = array.positions[:, 1]
    r_n = np.sqrt(
        (target[0] - x_n) ** 2 + (target[1] - y_n) ** 2 + target[2] ** 2
    )[:, None]  # (N, 1)
    amplitude = 1.0 / r_n
    phase = 2 * np.pi * center_freq / C * r_n
    a_single_freq = amplitude * np.exp(-1j * phase)
    a_single_freq = _normalize_energy(a_single_freq)

    F = np.atleast_1d(freqs).shape[0]
    return np.repeat(a_single_freq, F, axis=1)


def squint_aware_steering_vector_planar(
    array: PlanarArrayGeometry,
    theta: float,
    phi: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Squint-aware (TTD, angle-only, per-frequency phase) steering
    vector, 2D UPA version. Mirrors beamformers.squint_aware_steering_vector."""
    x_n = array.positions[:, 0][:, None]
    y_n = array.positions[:, 1][:, None]
    f = np.atleast_1d(freqs)[None, :]
    phase = 2 * np.pi * f / C * (x_n * np.sin(theta) * np.cos(phi) + y_n * np.sin(phi))
    return np.exp(1j * phase)  # unit modulus -> already sum|a_n|^2=N per freq column


def combined_steering_vector_planar(
    array: PlanarArrayGeometry,
    r: float,
    theta: float,
    phi: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Combined near-field + squint-aware corrector, 2D UPA version.
    Mirrors beamformers.combined_steering_vector (reuses the exact
    near-field channel model, then per-frequency energy-normalizes)."""
    a = near_field_channel_planar(array, r, theta, phi, freqs)
    return _normalize_energy(a)


def beamform_planar(
    mode: str,
    array: PlanarArrayGeometry,
    freqs: np.ndarray,
    theta: float,
    phi: float,
    r: float = None,
    center_freq: float = None,
) -> np.ndarray:
    """Uniform dispatch across all four modes, 2D UPA version. Mirrors
    beamformers.beamform's interface with phi added."""
    if mode == "conventional":
        if center_freq is None:
            raise ValueError("conventional mode requires center_freq")
        return conventional_steering_vector_planar(array, theta, phi, freqs, center_freq)
    elif mode == "nearfield":
        if r is None or center_freq is None:
            raise ValueError("nearfield mode requires r and center_freq")
        return nearfield_steering_vector_planar(array, r, theta, phi, freqs, center_freq)
    elif mode == "squint":
        return squint_aware_steering_vector_planar(array, theta, phi, freqs)
    elif mode == "combined":
        if r is None:
            raise ValueError("combined mode requires r")
        return combined_steering_vector_planar(array, r, theta, phi, freqs)
    else:
        raise ValueError(f"Unknown beamformer mode: {mode!r}")