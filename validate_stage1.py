"""
Stage 1 checkpoint script.

Confirms:
    1. Array + signal generator runs correctly and quickly at N=64.
    2. Profiles a single "channel-computation-shaped" vectorized op
       (per-element x per-frequency phase term, as near-field/squint models
       will need in Stage 2+) across N = 8 -> 512, checking for
       near-linear-per-frequency scaling rather than blowup.
    3. Sets the real N ceiling for the rest of the project based on that
       profiling result.

Run: python3 validate_stage1.py
"""

import time

import numpy as np

from arraymodel import C, generate_setup

CENTER_FREQ = 28e9  # 28 GHz, upper-midband-ish 6G reference point
BANDWIDTH = 400e6


def time_setup(N: int, n_reps: int = 5) -> float:
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        setup = generate_setup(N=N, bandwidth=BANDWIDTH, center_freq=CENTER_FREQ)
        _ = setup["array"].positions.sum()  # touch the array to force materialization
        _ = setup["signal"]["symbols"].sum()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)  # min of reps to reduce noise from scheduling jitter


def time_vectorized_channel_shape(N: int, n_freqs: int = 64, n_reps: int = 5) -> float:
    """Profile the shape of computation Stage 2+ will actually do:
    a full (N elements x F frequencies) complex exponential, built with
    pure NumPy broadcasting, no Python loops over elements or frequencies.
    """
    setup = generate_setup(N=N, bandwidth=BANDWIDTH, center_freq=CENTER_FREQ)
    array = setup["array"]
    freqs = CENTER_FREQ + np.linspace(-BANDWIDTH / 2, BANDWIDTH / 2, n_freqs)

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        # (N, 1) and (1, F) broadcast to (N, F) -- this is the same broadcasting
        # pattern the near-field / squint-aware channel models will use.
        d_n = array.positions[:, None]          # (N, 1)
        f = freqs[None, :]                      # (1, F)
        r = np.sqrt(d_n**2 + 1000.0**2)          # dummy distance term, vectorized
        phase_term = np.exp(-1j * 2 * np.pi * f / C * r)
        _ = phase_term.sum()  # force evaluation
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)


def main():
    print("=== Stage 1 Checkpoint ===\n")

    print("-- Correctness / speed check at N=64 --")
    t64 = time_setup(64)
    print(f"generate_setup(N=64): {t64*1000:.3f} ms (should be << 1s)")
    assert t64 < 1.0, "N=64 setup should take well under a second"

    setup64 = generate_setup(N=64, bandwidth=BANDWIDTH, center_freq=CENTER_FREQ)
    arr64 = setup64["array"]
    sig64 = setup64["signal"]
    print(f"  N={arr64.N}, spacing={arr64.spacing*1000:.3f} mm, "
          f"aperture={arr64.aperture:.3f} m")
    print(f"  Rayleigh distance @ N=64: {arr64.rayleigh_distance():.2f} m")
    print(f"  n_subcarriers={sig64['freqs'].shape[0]}, "
          f"symbols shape={sig64['symbols'].shape}")

    # Sanity: aperture should scale linearly with N (spacing fixed).
    setup128 = generate_setup(N=128, bandwidth=BANDWIDTH, center_freq=CENTER_FREQ)
    ratio = setup128["array"].aperture / arr64.aperture
    print(f"  aperture(128)/aperture(64) = {ratio:.3f} (expect ~2.0 as N doubles)")
    assert 1.9 < ratio < 2.1, "aperture should scale ~linearly with N"

    print("\n-- Profiling across N (channel-computation-shaped op) --")
    Ns = [8, 32, 64, 128, 256, 512]
    times = {}
    for N in Ns:
        t = time_vectorized_channel_shape(N)
        times[N] = t
        print(f"  N={N:4d}: {t*1000:8.4f} ms")

    print("\n-- Scaling check (64 -> 512, expect ~linear i.e. ~8x) --")
    t64c = times[64]
    t512c = times[512]
    scale_factor = t512c / t64c
    print(f"  512/64 array-size ratio: 8.0x")
    print(f"  512/64 time ratio:       {scale_factor:.2f}x")

    if scale_factor > 8.0 * 4:  # more than ~4x worse than expected linear scaling
        print("  WARNING: scaling is much worse than linear -- possible "
              "vectorization bug or O(N^2) op. Investigate before Stage 2.")
        ceiling = 256
    else:
        print("  OK: scaling is roughly linear, no red flags.")
        ceiling = 512

    print(f"\n=== Real N ceiling for this project: {ceiling} ===")
    print("(1024 may be attempted later only if time/perf allow -- not required)")


if __name__ == "__main__":
    main()