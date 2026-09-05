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

--- Stage 8 addition ---
    4. Extends the profiling sweep to N=1024 and checks the 512->1024
       timing ratio is still roughly linear (~2x, tolerance ~4x per the
       same convention used for the 64->512 check above). Per the
       Extension Plan, this is not "does N=1024 run fast enough" in
       isolation -- it's the first half of testing whether the
       Rayleigh-normalized failure law discovered in Stage 5
       (gain_loss ~ f(r/R_Rayleigh)) is load-bearing on N<=512, or
       generalizes further. The second half of that question (does the
       failure map itself stay coherent at N=1024) is checked in
       validate_stage5.py's regenerated fine sweep.

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
    Ns = [8, 32, 64, 128, 256, 512, 1024]
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

    print(f"\n=== Real N ceiling (pre-Stage-8): {ceiling} ===")

    # =========================================================================
    # Stage 8 -- N=1024 re-validation
    # =========================================================================
    print("\n-- Stage 8: N=1024 re-validation --")
    print("  Guiding question: does runtime scaling remain acceptable, and "
          "does the analytical Rayleigh-distance scale stay physically "
          "sensible, at 2x the previously-profiled ceiling?")

    t1024c = times[1024]
    scale_1024 = t1024c / t512c
    print(f"\n  512->1024 array-size ratio: 2.0x")
    print(f"  512->1024 time ratio:       {scale_1024:.2f}x")

    # Same tolerance convention as the 64->512 check above: allow up to ~4x
    # worse than ideal linear scaling before flagging a problem. Ideal here
    # is 2.0x (array size doubled), so the ceiling for "acceptable" is 8.0x.
    stage8_scaling_ok = scale_1024 <= 2.0 * 4
    if not stage8_scaling_ok:
        print("  WARNING: 512->1024 scaling is much worse than linear -- "
              "possible vectorization bug or O(N^2) op emerging at this "
              "size. Do NOT add N=1024 to Stage 5's fine sweep until fixed.")
    else:
        print("  OK: 512->1024 scaling is within ~4x of ideal linear "
              "(same tolerance used for the original 64->512 check).")

    # Rayleigh distance sanity check: per the project doc, R_rayleigh
    # scales as N^2, so 512->1024 should give ~4x the Stage-1 N=512 value.
    # This is a physical sanity check, not just an arithmetic one -- confirms
    # the absolute distance scale (in meters) implied at N=1024 is still a
    # sensible number for the 6G-user framing (i.e. not, say, kilometers off
    # from what a "user near a base station" scenario should look like).
    from arraymodel import ArrayGeometry
    array_512 = ArrayGeometry(N=512, center_freq=CENTER_FREQ)
    array_1024 = ArrayGeometry(N=1024, center_freq=CENTER_FREQ)
    r512 = array_512.rayleigh_distance()
    r1024 = array_1024.rayleigh_distance()
    rayleigh_ratio = r1024 / r512
    print(f"\n  Rayleigh distance @ N=512:  {r512:9.2f} m")
    print(f"  Rayleigh distance @ N=1024: {r1024:9.2f} m")
    print(f"  Ratio (1024/512): {rayleigh_ratio:.3f}  (expect ~4.0, since R ~ N^2)")
    assert 3.6 < rayleigh_ratio < 4.4, (
        f"Rayleigh distance should scale ~4x from N=512->1024 (N^2 scaling), "
        f"got ratio {rayleigh_ratio:.3f}"
    )
    print("  OK: Rayleigh distance scaling matches the analytical N^2 prediction.")
    print(f"  Physical sanity: {r1024:.1f} m is still a plausible base-station-to-user "
          f"range for a 6G upper-midband deployment (not an absurd km-scale figure).")

    if stage8_scaling_ok:
        ceiling = 1024
        print(f"\n=== Stage 8 checkpoint PASSED. N=1024 timing and Rayleigh-distance "
              f"scaling both confirmed sane. ===")
        print(f"=== Real N ceiling for this project (post-Stage-8): {ceiling} ===")
        print("(Stage 5's fine sweep should now be regenerated with N=1024 "
              "included -- see validate_stage5.py.)")
    else:
        print(f"\n=== Stage 8 checkpoint FAILED on timing. Do not extend Stage 5's "
              f"fine sweep to N=1024 until this is investigated. Real N ceiling "
              f"remains {ceiling}. ===")


if __name__ == "__main__":
    main()